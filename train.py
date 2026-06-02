"""
训练管线

流程：
  1. 加载数据，构建 StockData 和 target
  2. 初始化 AlphaCombinationModel 和 PPO Agent
  3. 循环：生成 episode → 评估 alpha → 更新组合模型 → PPO 更新
  4. 保存最终 alpha 池
"""

import os
import json
import time
import numpy as np
import torch

from tokens import (
    VOCAB,
    VOCAB_SIZE,
    TOKEN_TO_IDX,
    TIME_DELTAS,
    Token,
    TokenType,
    BEG_TOKEN,
    SEP_TOKEN,
)
from expression import (
    parse_rpn_to_tree,
    strip_special_tokens,
    tree_to_formula,
    parse_formula,
)
from calculator import StockData
from combination import AlphaCombinationModel
from masking import RPNBuilder
from generator import AlphaGenNet, PPOAgent, GRPOAgent, Episode
from data import Data
from common import ADJUST_PREV
from reward import calculate_enhanced_reward, set_reward_mode
from reporting import (
    CombinationEvaluator,
    WarmupDataContext,
    compute_benchmark_result,
    evaluate_weighted_alpha_values,
    estimate_required_warmup_days,
    plot_backtest_comparison,
    plot_equity_curves,
    plot_training_history,
    save_json,
    save_training_history,
)
from backtest import Backtester, load_pool_alpha_from_file

BEG_IDX = TOKEN_TO_IDX[Token(TokenType.BEG, BEG_TOKEN)]
SEP_IDX = TOKEN_TO_IDX[Token(TokenType.SEP, SEP_TOKEN)]


# ============================================================
# 1. 单 episode 生成
# ============================================================
def collect_episode(
    agent: PPOAgent | GRPOAgent,
    builder: RPNBuilder,
    device: str = "cpu",
) -> Episode:
    """生成一条 RPN 序列。"""
    builder.reset()
    ep = Episode()

    # 起始 BEG
    builder.step(BEG_IDX)
    ep.token_ids.append(BEG_IDX)

    hidden = agent.net.init_hidden(batch_size=1, device=device)

    while not builder.done:
        valid_mask = builder.get_valid_mask()

        if not valid_mask.any():
            break

        action, log_prob, value, hidden = agent.select_action(
            ep.token_ids[-1], valid_mask, hidden
        )

        ep.actions.append(action)
        ep.log_probs.append(log_prob)
        ep.values.append(value)
        ep.masks.append(valid_mask.copy())

        builder.step(action)
        ep.token_ids.append(action)

    return ep


# ============================================================
# 2. 评估 episode 并计算 reward
# ============================================================
def evaluate_episode(
    ep: Episode,
    combination: AlphaCombinationModel,
) -> dict:
    """
    解析 episode 的 token 序列，评估 alpha，加入组合模型。
    返回结构化结果：
      - reward: PPO 使用的奖励
      - accepted: 新 alpha 是否被接纳进组合
      - formula: 表达式字符串（若可解析）
      - candidate_ic: 该 alpha 单独对 target 的 IC
      - ic_delta: 接纳后带来的组合 IC 增量
      - status: 接纳/拒绝原因
    """

    def _result(
        reward: float,
        accepted: bool = False,
        formula: str | None = None,
        candidate_ic: float | None = None,
        ic_delta: float | None = None,
        status: str = "invalid",
    ) -> dict:
        return {
            "reward": float(reward),
            "accepted": bool(accepted),
            "formula": formula,
            "candidate_ic": candidate_ic,
            "ic_delta": ic_delta,
            "status": status,
        }

    tokens = [VOCAB[i] for i in ep.token_ids]
    stripped = strip_special_tokens(tokens)

    if len(stripped) == 0:
        return _result(-1.0, status="invalid_empty")

    tree = parse_rpn_to_tree(stripped)
    if tree is None:
        return _result(-1.0, status="invalid_parse")

    formula = tree_to_formula(tree)

    try:
        alpha_val = combination.calculator.evaluate(tree)
    except Exception:
        return _result(-1.0, formula=formula, status="invalid_eval_exception")

    if alpha_val is None or np.all(np.isnan(alpha_val)):
        return _result(-1.0, formula=formula, status="invalid_all_nan")

    valid_ratio = np.mean(~np.isnan(alpha_val))
    if valid_ratio < 0.1:
        return _result(-1.0, formula=formula, status="invalid_sparse")

    existing_ics = combination.ic_vector.copy()
    old_ic = combination.get_combination_ic()
    new_ic = combination.add_alpha(
        tree,
        alpha_values=alpha_val,
        already_normed=True,
        baseline_ic=old_ic,
    )
    if combination.last_add_accepted:
        ic_delta = new_ic - old_ic
        reward = calculate_enhanced_reward(
            old_ic=old_ic,
            new_ic=new_ic,
            candidate_ic=combination.last_add_candidate_ic,
            candidate_ic_std=combination.last_add_ic_std,
            existing_ics=existing_ics,
            token_ids=ep.token_ids,
            vocab=VOCAB,
            accepted=True,
            status=combination.last_add_status,
        )
        return _result(
            reward=reward,
            accepted=True,
            formula=formula,
            candidate_ic=combination.last_add_candidate_ic,
            ic_delta=ic_delta,
            status=combination.last_add_status,
        )

    reward = calculate_enhanced_reward(
        old_ic=old_ic,
        new_ic=new_ic,
        candidate_ic=combination.last_add_candidate_ic,
        candidate_ic_std=combination.last_add_ic_std,
        existing_ics=existing_ics,
        token_ids=ep.token_ids,
        vocab=VOCAB,
        accepted=False,
        status=combination.last_add_status,
    )
    return _result(
        reward=reward,
        accepted=False,
        formula=formula,
        candidate_ic=combination.last_add_candidate_ic,
        ic_delta=combination.last_add_ic_delta,
        status=combination.last_add_status,
    )


# ============================================================
# 3. 主训练函数
# ============================================================
def train(
    stock_data: StockData,
    target: np.ndarray,
    val_stock_data: StockData | None = None,
    val_target: np.ndarray | None = None,
    val_data_context: WarmupDataContext | None = None,
    num_iterations: int = 100,
    episodes_per_iter: int = 64,
    max_pool_size: int = 20,
    max_seq_len: int = 20,
    horizon: int = 20,
    lr: float = 3e-4,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    algo: str = "ppo",
    model_type: str = "transformer",
    patience: int = 5,
    use_enhanced_reward: bool = True,
):
    """Algorithm 2 完整训练循环。支持 PPO/GRPO 与 early stopping。"""
    os.makedirs(save_dir, exist_ok=True)
    set_reward_mode(use_enhanced_reward)

    # 初始化组合模型
    combination = AlphaCombinationModel(stock_data, target, max_pool_size=max_pool_size)

    # 初始化网络
    model_type = model_type.lower()
    net = AlphaGenNet(vocab_size=VOCAB_SIZE, model_type=model_type)
    algo = algo.lower()
    if algo == "grpo":
        agent = GRPOAgent(net, lr=lr, device=device)
        print("训练算法: GRPO")
    else:
        agent = PPOAgent(net, lr=lr, device=device)
        print("训练算法: PPO")
    print(f"生成器结构: {model_type.upper()}")
    print(
        "奖励模式: "
        + ("增强版多目标奖励" if use_enhanced_reward else "基础单奖励(IC增量)")
    )

    builder = RPNBuilder(max_len=max_seq_len)
    val_evaluator = (
        CombinationEvaluator(val_stock_data, val_target)
        if val_stock_data is not None
        and val_target is not None
        and val_data_context is None
        else None
    )
    preloaded_val_warmup = None
    if val_data_context is not None:
        preloaded_val_warmup = val_data_context.preload_generator_warmup(
            max_seq_len=max_seq_len,
            time_deltas=TIME_DELTAS,
        )
    best_score = -np.inf
    no_improve_count = 0
    history = []

    print(f"开始训练: {num_iterations} 轮, 每轮 {episodes_per_iter} episode")
    print(f"设备: {device}, 股票数: {stock_data.n_stocks}, 交易日: {stock_data.n_days}")
    print("股票过滤: 默认排除北交所(BJ)、ST股票、上市未满1年的新股")
    if val_data_context is not None:
        print(
            f"验证集: 股票数 {val_data_context.stock_data.n_stocks}, "
            f"交易日 {len(val_data_context.official_trade_dates())}, "
            f"正式起点 {val_data_context.start_date_str}"
        )
        print(
            f"验证预热: 预加载 {preloaded_val_warmup} 个交易日, "
            f"扩展起点 {val_data_context.loaded_start_date_str}"
        )
    elif val_stock_data is not None:
        print(
            f"验证集: 股票数 {val_stock_data.n_stocks}, 交易日 {val_stock_data.n_days}"
        )
    print("=" * 70)

    for iteration in range(num_iterations):
        t0 = time.time()
        episodes = []
        rewards = []
        valid_count = 0
        accepted_alphas = []

        # 收集 episodes
        for _ in range(episodes_per_iter):
            ep = collect_episode(agent, builder, device=device)

            if len(ep.actions) == 0:
                continue

            episode_result = evaluate_episode(ep, combination)
            reward = episode_result["reward"]
            ep.reward = reward
            episodes.append(ep)
            rewards.append(reward)

            if reward > -0.5:
                valid_count += 1
            if episode_result["accepted"]:
                accepted_alphas.append(
                    {
                        "formula": episode_result["formula"],
                        "candidate_ic": float(episode_result["candidate_ic"]),
                        "ic_delta": float(episode_result["ic_delta"]),
                    }
                )

        # PPO 更新
        stats = agent.update(episodes)

        # 日志
        elapsed = time.time() - t0
        avg_reward = np.mean(rewards) if rewards else 0
        avg_len = np.mean([len(ep.token_ids) for ep in episodes]) if episodes else 0
        train_metrics = evaluate_weighted_alpha_values(
            combination.alpha_values,
            combination.weights,
            target,
            already_normed_target=False,
        )
        val_metrics = (
            val_data_context.evaluate(combination.alpha_exprs, combination.weights)
            if val_data_context is not None
            else (
                val_evaluator.evaluate(combination.alpha_exprs, combination.weights)
                if val_evaluator is not None
                else None
            )
        )
        selection_score = (
            val_metrics["ic"] if val_metrics is not None else train_metrics["ic"]
        )
        if not np.isfinite(selection_score):
            selection_score = -np.inf
        val_log = ""
        if val_metrics is not None:
            val_log = (
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_ic={val_metrics['ic']:+.4f}  "
                f"val_icir={val_metrics['icir']:+.4f}  "
            )

        row = {
            "iteration": iteration + 1,
            "pool_size": combination.pool_size,
            "accepted_alpha_count": int(len(accepted_alphas)),
            "accepted_alphas": accepted_alphas,
            "train_loss": train_metrics["loss"],
            "train_ic": train_metrics["ic"],
            "train_icir": train_metrics["icir"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_ic": None if val_metrics is None else val_metrics["ic"],
            "val_icir": None if val_metrics is None else val_metrics["icir"],
            "avg_reward": float(avg_reward),
            "valid_episodes": int(valid_count),
            "num_episodes": int(len(episodes)),
            "avg_seq_len": float(avg_len),
            "policy_loss": float(stats.get("policy_loss", 0.0)),
            "value_loss": float(stats.get("value_loss", 0.0)),
            "entropy": float(stats.get("entropy", 0.0)),
            "elapsed_sec": float(elapsed),
        }
        history.append(row)

        print(
            f"[{iteration+1:3d}/{num_iterations}] "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_ic={train_metrics['ic']:+.4f}  "
            f"train_icir={train_metrics['icir']:+.4f}  "
            f"{val_log}"
            f"avg_r={avg_reward:+.4f}  "
            f"accepted={len(accepted_alphas)}  "
            f"pool={combination.pool_size}  "
            f"valid={valid_count}/{len(episodes)}  "
            f"avg_len={avg_len:.1f}  "
            f"p_loss={stats.get('policy_loss', 0):.4f}  "
            f"v_loss={stats.get('value_loss', 0):.4f}  "
            f"entropy={stats.get('entropy', 0):.3f}  "
            f"time={elapsed:.1f}s"
        )
        if accepted_alphas:
            for idx, info in enumerate(accepted_alphas, start=1):
                print(
                    f"    accepted[{idx}] "
                    f"candidate_ic={info['candidate_ic']:+.4f}  "
                    f"ic_delta={info['ic_delta']:+.4f}  "
                    f"{info['formula']}"
                )

        # 保存最佳
        if selection_score > best_score + 1e-5 and combination.pool_size > 0:
            best_score = selection_score
            no_improve_count = 0
            save_checkpoint(combination, net, save_dir, tag="best")
        else:
            no_improve_count += 1

        if patience > 0 and no_improve_count >= patience:
            print(
                f"Early stopping at iteration {iteration + 1} "
                f"(no improvement for {patience} iterations)"
            )
            break

        # 每 20 轮保存
        if (iteration + 1) % 20 == 0:
            save_checkpoint(combination, net, save_dir, tag=f"iter{iteration+1}")

    # 最终保存
    save_checkpoint(combination, net, save_dir, tag="final")
    save_training_history(history, save_dir)
    plot_training_history(history, os.path.join(save_dir, "training_metrics.png"))

    final_train_metrics = evaluate_weighted_alpha_values(
        combination.alpha_values,
        combination.weights,
        target,
        already_normed_target=False,
    )
    final_val_metrics = (
        val_data_context.evaluate(combination.alpha_exprs, combination.weights)
        if val_data_context is not None
        else (
            val_evaluator.evaluate(combination.alpha_exprs, combination.weights)
            if val_evaluator is not None
            else None
        )
    )
    summary = {
        "algorithm": algo,
        "model_type": model_type,
        "patience": patience,
        "use_enhanced_reward": bool(use_enhanced_reward),
        "best_selection_score": float(best_score) if np.isfinite(best_score) else None,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "history_length": len(history),
        "stopped_early": len(history) < num_iterations,
    }
    save_json(summary, os.path.join(save_dir, "training_summary.json"))

    print("=" * 70)
    print(
        f"训练完成. 最终 train_IC = {final_train_metrics['ic']:+.4f}, "
        f"train_ICIR = {final_train_metrics['icir']:+.4f}"
    )
    if final_val_metrics is not None:
        print(
            f"最终 val_IC = {final_val_metrics['ic']:+.4f}, "
            f"val_ICIR = {final_val_metrics['icir']:+.4f}"
        )
    combination.summary()

    return combination, net, history


def save_checkpoint(combination, net, save_dir, tag="latest"):
    # 保存网络
    torch.save(net.state_dict(), os.path.join(save_dir, f"net_{tag}.pt"))

    # 保存 alpha 池信息
    pool_info = []
    for i, expr in enumerate(combination.alpha_exprs):
        pool_info.append(
            {
                "formula": tree_to_formula(expr),
                "weight": float(combination.weights[i]),
                "ic": float(combination.ic_vector[i]),
            }
        )
    with open(os.path.join(save_dir, f"pool_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(pool_info, f, ensure_ascii=False, indent=2)


# ============================================================
# 4. 入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_start", default="20190101")
    parser.add_argument("--train_end", default="20241231")
    parser.add_argument("--val_start", default="20250101")
    parser.add_argument("--val_end", default="20251231")
    parser.add_argument("--test_start", default="20260101")
    parser.add_argument("--test_end", default="20260501")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--pool_size", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--n_hold", type=int, default=20)
    parser.add_argument("--n_swap", type=int, default=3)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--benchmark_code", default="000300.SH")
    parser.add_argument(
        "--algo",
        choices=["ppo", "grpo"],
        default="ppo",
        help="训练算法: ppo (默认) 或 grpo",
    )
    parser.add_argument(
        "--model_type",
        choices=["lstm", "transformer"],
        default="lstm",
        help="生成器结构: lstm (默认) 或 transformer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="连续多少轮验证分数未提升后提前停止；<=0 表示关闭",
    )
    parser.add_argument(
        "--use_enhanced_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用增强版多目标奖励；关闭后回退到基础 IC 增量奖励",
    )
    parser.add_argument("--save_dir", default="outputs")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("加载训练集数据...")
    reader = Data()
    train_df = reader.daily(
        start_date=args.train_start,
        end_date=args.train_end,
        bj=False,
        st=False,
        adjust=ADJUST_PREV,
    )

    train_df = train_df[
        ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "vwap"]
    ]
    print(f"训练集数据行数: {len(train_df)}")

    train_sd = StockData(train_df)
    train_target = train_sd.get_target(horizon=args.horizon)

    print("加载验证集数据...")
    val_context = WarmupDataContext(
        reader=reader,
        start_date=args.val_start,
        end_date=args.val_end,
        horizon=args.horizon,
        adjust=ADJUST_PREV,
        bj=False,
        st=False,
    )
    print("验证集数据将在训练开始前按生成器最大窗口一次性预热加载。")

    combination, net, _ = train(
        stock_data=train_sd,
        target=train_target,
        val_data_context=val_context,
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        max_pool_size=args.pool_size,
        horizon=args.horizon,
        device=args.device,
        save_dir=args.save_dir,
        algo=args.algo,
        model_type=args.model_type,
        patience=args.patience,
        use_enhanced_reward=args.use_enhanced_reward,
    )

    pool_file = os.path.join(args.save_dir, "pool_best.json")
    if os.path.isfile(pool_file):
        with open(pool_file, "r", encoding="utf-8") as f:
            pool_info = json.load(f)
        pool_exprs = []
        for item in pool_info:
            tree = parse_formula(item["formula"])
            if tree is not None:
                pool_exprs.append(tree)
        warmup_days = estimate_required_warmup_days(pool_exprs)

        # 加载测试集数据
        print("\n加载测试集数据...")
        test_context = WarmupDataContext(
            reader=reader,
            start_date=args.test_start,
            end_date=args.test_end,
            horizon=args.horizon,
            adjust=ADJUST_PREV,
            bj=False,
            st=False,
        )
        test_context.ensure_warmup_days(warmup_days)
        print(
            f"测试集 warm-up: {warmup_days} 个交易日, "
            f"扩展起点 = {test_context.loaded_start_date_str}, "
            f"正式统计起点 = {args.test_start}"
        )

        pool_state = load_pool_alpha_from_file(
            pool_file,
            test_context.stock_data,
            target=test_context.target,
            verbose=True,
            official_start_index=test_context.official_start_index,
        )
        combo_alpha = pool_state["combo_alpha"][test_context.official_start_index :]
        factor_metrics = pool_state["factor_metrics"]
        print(
            f"\n最佳测试池因子表现: IC={factor_metrics['ic']:+.4f}, "
            f"ICIR={factor_metrics['icir']:+.4f}, Loss={factor_metrics['loss']:.4f}"
        )

        bt = Backtester(
            n_hold=args.n_hold,
            n_swap=args.n_swap,
            commission=args.commission,
        )
        official_trade_dates = test_context.official_trade_dates()
        backtest_eligibility_mask = reader.seasoned_mask(
            official_trade_dates,
            test_context.stock_data.stock_codes,
            min_list_days=365,
        )
        combo_alpha_for_backtest = np.where(
            backtest_eligibility_mask, combo_alpha, np.nan
        )
        strategy_result = bt.run(
            combo_alpha_for_backtest,
            test_context.official_open_prices(),
            official_trade_dates,
            test_context.stock_data.stock_codes,
        )
        benchmark_result = compute_benchmark_result(
            reader.market(args.benchmark_code),
            official_trade_dates,
            start_offset=strategy_result["trade_date_offset"],
        )

        print(f"\n回测结果（测试集，对比 {args.benchmark_code}）")
        print("  回测买入限制: 排除上市未满1年的新股")
        print(
            f"  策略: 年化={strategy_result['annual_return']:+.2%}  "
            f"夏普={strategy_result['sharpe_ratio']:.3f}  "
            f"回撤={strategy_result['max_drawdown']:.2%}"
        )
        print(
            f"  基准: 年化={benchmark_result['annual_return']:+.2%}  "
            f"夏普={benchmark_result['sharpe_ratio']:.3f}  "
            f"回撤={benchmark_result['max_drawdown']:.2%}"
        )

        save_json(
            {
                "factor_metrics": factor_metrics,
                "strategy_backtest": strategy_result,
                "benchmark_backtest": benchmark_result,
                "benchmark_code": args.benchmark_code,
                "warmup_days": warmup_days,
                "warmup_start": test_context.loaded_start_date_str,
                "official_test_start": args.test_start,
                "official_test_end": args.test_end,
                "backtest_min_list_days": 365,
            },
            os.path.join(args.save_dir, "test_report.json"),
        )
        plot_backtest_comparison(
            strategy_result,
            benchmark_result,
            os.path.join(args.save_dir, "backtest_vs_benchmark.png"),
            benchmark_label=args.benchmark_code,
        )
        plot_equity_curves(
            strategy_result,
            benchmark_result,
            os.path.join(args.save_dir, "equity_curve_vs_benchmark.png"),
            benchmark_label=args.benchmark_code,
        )


if __name__ == "__main__":
    main()
