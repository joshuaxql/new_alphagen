"""
增强版奖励函数。

改进点：
  - 多目标奖励：IC 增量、ICIR、正负因子多样性
  - 表达式复杂度惩罚
  - 保留开关，可回退到原始奖励
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import REWARD_CONFIG
from tokens import TokenType


@dataclass
class RewardBreakdown:
    total_reward: float
    ic_reward: float
    icir_reward: float
    diversity_reward: float
    complexity_penalty: float


class EnhancedRewardCalculator:
    def __init__(
        self,
        ic_weight: float | None = None,
        icir_weight: float | None = None,
        diversity_weight: float | None = None,
        complexity_penalty_factor: float | None = None,
        max_complexity: int | None = None,
    ):
        self.ic_weight = (
            ic_weight if ic_weight is not None else REWARD_CONFIG["ic_weight"]
        )
        self.icir_weight = (
            icir_weight if icir_weight is not None else REWARD_CONFIG["icir_weight"]
        )
        self.diversity_weight = (
            diversity_weight
            if diversity_weight is not None
            else REWARD_CONFIG["diversity_weight"]
        )
        self.complexity_penalty_factor = (
            complexity_penalty_factor
            if complexity_penalty_factor is not None
            else REWARD_CONFIG["complexity_penalty_factor"]
        )
        self.max_complexity = (
            max_complexity
            if max_complexity is not None
            else REWARD_CONFIG["max_complexity"]
        )

    def calculate_reward_with_breakdown(
        self,
        old_ic: float,
        new_ic: float,
        candidate_ic: float,
        candidate_ic_std: float,
        existing_ics: np.ndarray,
        token_ids: List[int],
        vocab,
        accepted: bool = True,
        status: str = "accepted",
    ) -> RewardBreakdown:
        if not accepted:
            penalty = self._get_rejection_penalty(status)
            return RewardBreakdown(
                total_reward=penalty,
                ic_reward=0.0,
                icir_reward=0.0,
                diversity_reward=0.0,
                complexity_penalty=0.0,
            )

        ic_delta = new_ic - old_ic
        ic_reward = self.ic_weight * ic_delta
        icir_reward = self.icir_weight * self._calculate_icir(
            candidate_ic, candidate_ic_std
        )
        diversity_reward = self._calculate_diversity_reward(
            candidate_ic, existing_ics
        )
        complexity = self._calculate_complexity(token_ids, vocab)
        complexity_penalty = self._calculate_complexity_penalty(complexity)

        total_reward = (
            ic_reward + icir_reward + diversity_reward - complexity_penalty
        )
        return RewardBreakdown(
            total_reward=total_reward,
            ic_reward=ic_reward,
            icir_reward=icir_reward,
            diversity_reward=diversity_reward,
            complexity_penalty=complexity_penalty,
        )

    def calculate_reward(
        self,
        old_ic: float,
        new_ic: float,
        candidate_ic: float,
        candidate_ic_std: float,
        existing_ics: np.ndarray,
        token_ids: List[int],
        vocab,
        accepted: bool = True,
        status: str = "accepted",
    ) -> float:
        breakdown = self.calculate_reward_with_breakdown(
            old_ic=old_ic,
            new_ic=new_ic,
            candidate_ic=candidate_ic,
            candidate_ic_std=candidate_ic_std,
            existing_ics=existing_ics,
            token_ids=token_ids,
            vocab=vocab,
            accepted=accepted,
            status=status,
        )
        return breakdown.total_reward

    @staticmethod
    def _calculate_icir(ic_mean: float, ic_std: float) -> float:
        if ic_std < 1e-12:
            return 0.0
        return ic_mean / ic_std

    def _calculate_diversity_reward(
        self, candidate_ic: float, existing_ics: np.ndarray
    ) -> float:
        if len(existing_ics) == 0:
            return 0.0

        pos_count = int(np.sum(existing_ics > 0))
        neg_count = int(np.sum(existing_ics < 0))
        total = len(existing_ics)
        if total == 0:
            return 0.0

        bias = abs(pos_count - neg_count) / total
        new_sign = np.sign(candidate_ic) if candidate_ic != 0 else 0
        if new_sign == 0:
            return 0.0
        if pos_count > neg_count and new_sign < 0:
            return self.diversity_weight * bias
        if neg_count > pos_count and new_sign > 0:
            return self.diversity_weight * bias
        return 0.0

    @staticmethod
    def _calculate_complexity(token_ids: List[int], vocab) -> int:
        length = len(token_ids)
        num_operators = 0
        for idx in token_ids:
            token = vocab[idx]
            if token.token_type == TokenType.OPERATOR:
                num_operators += 1
        return length + num_operators * 2

    def _calculate_complexity_penalty(self, complexity: int) -> float:
        if complexity <= self.max_complexity:
            return 0.0
        excess = complexity - self.max_complexity
        return self.complexity_penalty_factor * excess

    @staticmethod
    def _get_rejection_penalty(status: str) -> float:
        rejection_penalty = {
            "rejected_low_ic": -0.2,
            "rejected_redundant": -0.15,
            "rejected_no_improve": -0.1,
            "invalid_empty": -1.0,
            "invalid_parse": -1.0,
            "invalid_eval_exception": -1.0,
            "invalid_all_nan": -1.0,
            "invalid_sparse": -1.0,
        }
        return rejection_penalty.get(status, -1.0)


REWARD_CALCULATOR = EnhancedRewardCalculator()


def set_reward_mode(use_enhanced_reward: bool) -> None:
    REWARD_CONFIG["use_enhanced_reward"] = bool(use_enhanced_reward)


def is_enhanced_reward_enabled() -> bool:
    return bool(REWARD_CONFIG["use_enhanced_reward"])


def calculate_enhanced_reward(
    old_ic: float,
    new_ic: float,
    candidate_ic: float,
    candidate_ic_std: float,
    existing_ics: np.ndarray,
    token_ids: List[int],
    vocab,
    accepted: bool = True,
    status: str = "accepted",
) -> float:
    if not is_enhanced_reward_enabled():
        if accepted:
            return (new_ic - old_ic) * 10.0
        return EnhancedRewardCalculator._get_rejection_penalty(status)

    return REWARD_CALCULATOR.calculate_reward(
        old_ic=old_ic,
        new_ic=new_ic,
        candidate_ic=candidate_ic,
        candidate_ic_std=candidate_ic_std,
        existing_ics=existing_ics,
        token_ids=token_ids,
        vocab=vocab,
        accepted=accepted,
        status=status,
    )


def calculate_enhanced_reward_with_breakdown(
    old_ic: float,
    new_ic: float,
    candidate_ic: float,
    candidate_ic_std: float,
    existing_ics: np.ndarray,
    token_ids: List[int],
    vocab,
    accepted: bool = True,
    status: str = "accepted",
) -> RewardBreakdown:
    return REWARD_CALCULATOR.calculate_reward_with_breakdown(
        old_ic=old_ic,
        new_ic=new_ic,
        candidate_ic=candidate_ic,
        candidate_ic_std=candidate_ic_std,
        existing_ics=existing_ics,
        token_ids=token_ids,
        vocab=vocab,
        accepted=accepted,
        status=status,
    )
