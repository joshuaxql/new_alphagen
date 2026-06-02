# AlphaGen 项目说明

本仓库将四位同学对 Alpha 因子生成项目的优化整合到了 `new_alphagen/` 中。整体流程保持为“表达式生成 -> 因子评估 -> Alpha 组合 -> 回测验证”，同时在训练算法、模型结构、奖励函数和数据/实验流程上做了增强。

## 代码框架

```text
new_alphagen/
├─ README.md              # 项目说明文档
├─ requirements.txt       # Python 依赖
├─ train.py               # 训练总入口
├─ generator.py           # 生成器网络与 PPO / GRPO Agent
├─ masking.py             # RPN 合法动作掩码与序列构建
├─ tokens.py              # token、操作符、词表定义
├─ expression.py          # 表达式树、RPN 解析与公式互转
├─ calculator.py          # 因子计算、标准化与 IC 指标
├─ combination.py         # Alpha 池维护与组合权重优化
├─ reward.py              # 基础/增强奖励函数
├─ reporting.py           # 训练评估、历史保存与可视化
├─ backtest.py            # Top-k / Drop-n 回测
├─ data.py                # A 股数据读取与过滤
├─ common.py              # 公共路径与常量
└─ config.py              # 训练/奖励默认配置
```

- `new_alphagen/train.py`：训练总入口。负责加载数据、采样表达式、计算奖励、更新策略网络，并在验证集/测试集上评估结果。
- `new_alphagen/generator.py`：生成器与强化学习代理。`AlphaGenNet` 支持 `Transformer` 和 `LSTM` 两种结构，并实现了 `PPOAgent`、`GRPOAgent`。
- `new_alphagen/tokens.py`、`expression.py`、`masking.py`：定义表达式 token、RPN 语法树解析和合法动作掩码，保证生成公式可解析、可执行。
- `new_alphagen/calculator.py`：根据表达式计算因子值，并构造训练目标、IC 等基础指标。
- `new_alphagen/combination.py`：维护 alpha 池、组合权重和组合 IC，用于筛选候选因子并构建最终组合。
- `new_alphagen/reward.py`、`config.py`：奖励模块，支持基础奖励和增强版多目标奖励切换。
- `new_alphagen/reporting.py`：提供 warm-up 管理、因子评估、训练历史保存和可视化工具。
- `new_alphagen/data.py`、`common.py`：数据读取与预处理。默认读取 `D:/科大云盘/A股数据`，支持复权、BJ/ST 过滤以及新股过滤。
- `new_alphagen/backtest.py`：Top-k / Drop-n 回测模块，用于训练后的策略验证与独立回测。

## 四部分优化

四位同学的优化最终可以概括为下面四块：

1. 训练算法扩展  
在 `generator.py` 中新增 `GRPOAgent`，在 `train.py` 中加入 `--algo {ppo,grpo}`，可以在同一套框架下切换 PPO 与 GRPO。

2. 生成器结构扩展  
`AlphaGenNet` 现在支持 `Transformer` 和 `LSTM` 两种策略网络，`train.py` 中提供 `--model_type {transformer,lstm}` 选项，方便比较不同生成器结构。

3. 奖励函数增强  
新增 `reward.py` 多目标奖励逻辑，将组合 IC 增量、单因子 ICIR、多样性奖励和复杂度惩罚结合起来，同时保留基础 IC 增量奖励开关。

1. 加入early stopping
项目补充了 early stopping，防止过拟合。

## 运行方式

```bash
python train.py --algo grpo --model_type transformer
```

常用选项：

- `--algo ppo` 或 `--algo grpo`
- `--model_type transformer` 或 `--model_type lstm`
- `--use_enhanced_reward` 或 `--no-use_enhanced_reward`
- `--patience`：控制 early stopping

## 总结

整合后的代码以 `new_alphagen/` 为统一版本，既保留了原有 AlphaGen 的主流程，也吸收了四位同学在算法、结构、奖励和工程实现上的改进，形成了一套更完整、更方便实验对比的因子生成与回测框架。
