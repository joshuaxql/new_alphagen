"""
Alpha Generator — 可切换 Transformer / LSTM 的策略网络 + PPO/GRPO 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import List

from tokens import VOCAB_SIZE


# ============================================================
# 1. 网络结构
# ============================================================
class AlphaGenNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        model_type: str = "transformer",
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        head_dim: int = 64,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type not in {"transformer", "lstm"}:
            raise ValueError(
                f"Unsupported model_type={model_type!r}, expected 'transformer' or 'lstm'"
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if self.model_type == "transformer":
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            repr_dim = embed_dim
        else:
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            repr_dim = hidden_dim

        self.policy_head = nn.Sequential(
            nn.Linear(repr_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, vocab_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(repr_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    @staticmethod
    def _build_causal_mask(seq_len: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def _encode_transformer(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, t) → (batch, t, embed_dim)，应用 causal mask"""
        t = token_ids.shape[1]
        positions = torch.arange(t, device=token_ids.device)
        x = self.embedding(token_ids) + self.pos_embedding(positions)
        causal_mask = self._build_causal_mask(t, token_ids.device)
        return self.transformer(x, mask=causal_mask)

    def forward(self, token_idx: torch.Tensor, context):
        """
        单步推理（自回归生成用）。
        """
        if self.model_type == "transformer":
            if context is None:
                context = self.init_hidden(batch_size=token_idx.shape[0], device=token_idx.device)
            new_context = torch.cat([context, token_idx.unsqueeze(1)], dim=1)
            out = self._encode_transformer(new_context)
            h = out[:, -1, :]
            return self.policy_head(h), self.value_head(h).squeeze(-1), new_context

        x = self.embedding(token_idx).unsqueeze(1)
        out, hidden = self.lstm(x, context)
        h = out.squeeze(1)
        return self.policy_head(h), self.value_head(h).squeeze(-1), hidden

    def forward_sequence(self, token_ids: torch.Tensor, lengths: torch.Tensor):
        """
        批量前向（PPO 更新时用）。

        参数:
            token_ids: (batch, max_len) padding 后的序列
            lengths:   (batch,) 每条序列实际长度
        返回:
            all_logits: (batch, max_len, vocab_size)
            all_values: (batch, max_len)
        """
        if self.model_type == "lstm":
            x = self.embedding(token_ids)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                out,
                batch_first=True,
                total_length=token_ids.shape[1],
            )
            return self.policy_head(unpacked), self.value_head(unpacked).squeeze(-1)

        batch, max_len = token_ids.shape
        positions = torch.arange(max_len, device=token_ids.device)
        x = self.embedding(token_ids) + self.pos_embedding(positions)
        causal_mask = self._build_causal_mask(max_len, token_ids.device)
        # padding mask：超出实际长度的位置设为 True（被忽略）
        pad_mask = (
            torch.arange(max_len, device=token_ids.device).unsqueeze(0)
            >= lengths.to(token_ids.device).unsqueeze(1)
        )
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.policy_head(out), self.value_head(out).squeeze(-1)

    def init_hidden(self, batch_size: int = 1, device=None):
        """
        Transformer 返回空上下文张量，LSTM 返回 (h, c)。
        """
        if device is None:
            device = next(self.parameters()).device
        if self.model_type == "transformer":
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


# ============================================================
# 2. Episode 数据
# ============================================================
@dataclass
class Episode:
    token_ids: List[int] = field(default_factory=list)  # 完整序列含 BEG
    actions: List[int] = field(default_factory=list)  # tokens[1:]
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    reward: float = 0.0


# ============================================================
# 3. PPO Agent
# ============================================================
class PPOAgent:
    def __init__(
        self,
        net: AlphaGenNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def select_action(self, token_idx: int, valid_mask: np.ndarray, hidden):
        """
        单步动作选择。
        返回: action_idx, log_prob, value, new_hidden
        """
        self.net.eval()
        t = torch.tensor([token_idx], device=self.device)
        logits, value, hidden = self.net(t, hidden)

        mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        logits[0, ~mask_t] = float("-inf")

        dist = Categorical(logits=logits[0])
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item(), hidden

    def update(self, episodes: List[Episode]):
        """PPO 多 epoch 更新。"""
        if not episodes:
            return {}

        self.net.train()

        # 展平所有 step
        all_input_ids = []  # 每步输入 token
        all_actions = []
        all_old_logprobs = []
        all_returns = []  # γ=1, return = episode reward
        all_masks_np = []

        for ep in episodes:
            n_steps = len(ep.actions)
            for i in range(n_steps):
                all_input_ids.append(ep.token_ids[: i + 1])  # 截止当前的序列
                all_actions.append(ep.actions[i])
                all_old_logprobs.append(ep.log_probs[i])
                all_returns.append(ep.reward)
                all_masks_np.append(ep.masks[i])

        n_total = len(all_actions)
        if n_total == 0:
            return {}

        # 为批量前向做 padding
        max_seq = max(len(s) for s in all_input_ids)
        padded = torch.zeros(n_total, max_seq, dtype=torch.long, device=self.device)
        lengths = torch.zeros(n_total, dtype=torch.long)
        for i, seq in enumerate(all_input_ids):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)

        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)

        # 掩码 (n_total, vocab_size)
        masks_t = torch.tensor(
            np.stack(all_masks_np), dtype=torch.bool, device=self.device
        )

        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_total)
            for start in range(0, n_total, self.batch_size):
                idx = perm[start : start + self.batch_size]
                b_pad = padded[idx]
                b_len = lengths[idx]
                b_act = actions_t[idx]
                b_olp = old_lp[idx]
                b_ret = returns_t[idx]
                b_mask = masks_t[idx]

                # 前向：取每条序列最后一个有效位置的输出
                all_logits, all_values = self.net.forward_sequence(b_pad, b_len)
                # 取每条序列最后位置
                last_idx = (b_len - 1).long().to(self.device)
                batch_idx = torch.arange(len(idx), device=self.device)
                logits = all_logits[batch_idx, last_idx]  # (B, vocab)
                values = all_values[batch_idx, last_idx]  # (B,)

                logits = logits.masked_fill(~b_mask, float("-inf"))
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()

                # Advantage
                advantages = b_ret - values.detach()
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                else:
                    advantages = advantages.clone()

                # PPO clipped objective
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_ret)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                n_updates += 1

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates
        return stats


# ============================================================
# 4. GRPO Agent
# ============================================================
class GRPOAgent:
    """
    Group Relative Policy Optimization.

    与 PPO 的主要区别：
      - 不使用 value baseline
      - 直接在同一批 episode 内按 reward 做相对标准化
      - 仍保留 clipped policy objective 与 entropy regularization
    """

    def __init__(
        self,
        net: AlphaGenNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        grpo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.grpo_epochs = grpo_epochs
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def select_action(self, token_idx: int, valid_mask: np.ndarray, hidden):
        """
        单步动作选择。
        返回: action_idx, log_prob, dummy_value, new_hidden
        """
        self.net.eval()
        t = torch.tensor([token_idx], device=self.device)
        logits, _, hidden = self.net(t, hidden)

        mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        logits[0, ~mask_t] = float("-inf")

        dist = Categorical(logits=logits[0])
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), 0.0, hidden

    def update(self, episodes: List[Episode]):
        """GRPO 多 epoch 更新。"""
        if not episodes:
            return {}

        self.net.train()

        rewards = np.array([ep.reward for ep in episodes], dtype=np.float64)
        reward_mean = float(rewards.mean()) if len(rewards) > 0 else 0.0
        reward_std = float(rewards.std())
        if reward_std < 1e-8:
            reward_std = 1.0

        # 展平所有 step
        all_input_ids = []
        all_actions = []
        all_old_logprobs = []
        all_advantages = []
        all_masks_np = []

        for ep in episodes:
            n_steps = len(ep.actions)
            advantage = (ep.reward - reward_mean) / reward_std
            for i in range(n_steps):
                all_input_ids.append(ep.token_ids[: i + 1])
                all_actions.append(ep.actions[i])
                all_old_logprobs.append(ep.log_probs[i])
                all_advantages.append(advantage)
                all_masks_np.append(ep.masks[i])

        n_total = len(all_actions)
        if n_total == 0:
            return {}

        max_seq = max(len(s) for s in all_input_ids)
        padded = torch.zeros(n_total, max_seq, dtype=torch.long, device=self.device)
        lengths = torch.zeros(n_total, dtype=torch.long)
        for i, seq in enumerate(all_input_ids):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)

        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(
            all_advantages, dtype=torch.float32, device=self.device
        )
        masks_t = torch.tensor(
            np.stack(all_masks_np), dtype=torch.bool, device=self.device
        )

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.grpo_epochs):
            perm = torch.randperm(n_total)
            for start in range(0, n_total, self.batch_size):
                idx = perm[start : start + self.batch_size]
                b_pad = padded[idx]
                b_len = lengths[idx]
                b_act = actions_t[idx]
                b_olp = old_lp[idx]
                b_adv = advantages_t[idx]
                b_mask = masks_t[idx]

                all_logits, _ = self.net.forward_sequence(b_pad, b_len)
                last_idx = (b_len - 1).long().to(self.device)
                batch_idx = torch.arange(len(idx), device=self.device)
                logits = all_logits[batch_idx, last_idx]

                logits = logits.masked_fill(~b_mask, float("-inf"))
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()

                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                loss = policy_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["entropy"] += entropy.mean().item()
                n_updates += 1

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates
        return stats
