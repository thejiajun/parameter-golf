# Parameter Golf 挑战项目

## 项目概述
OpenAI Parameter Golf 挑战赛：在 16MB 大小限制内训练最好的语言模型。
评估指标：FineWeb 验证集上的 val_bpb（bits per byte），越低越好。
截止日期：2026-04-30（已结束，最终榜首 1.0565，见 README Leaderboard）。

## 关键约束
- 模型代码 + 压缩权重 ≤ 16,000,000 字节（16MB 十进制）
- 训练时间 ≤ 10 分钟（8×H100 SXM）
- 评估时间 ≤ 10 分钟（8×H100 SXM）
- 评估时不能访问训练数据或网络
- 新纪录需比当前 SOTA 好 ≥ 0.005 nats，p < 0.01

## 本地环境
- 仓库：直接 clone 自 `openai/parameter-golf`（origin 指向官方，不是 fork，无推送权限）
- 机器：Mac M4 Max
- Python 虚拟环境：`.venv/`
- 数据集：`data/datasets/fineweb10B_sp1024/`（已下载 1 shard）
- Tokenizer：`data/tokenizers/fineweb_1024_bpe.model`
- MLX 训练脚本：`train_gpt_mlx.py`（本地快速实验用）
- CUDA 训练脚本：`train_gpt.py`（正式提交用，需要 H100）

## 常用命令
```bash
# 激活环境
source .venv/bin/activate

# 本地 smoke test（200 步，几分钟）
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# 下载更多训练数据（完整 80 shards）
python3 data/cached_challenge_fineweb.py --variant sp1024

# 下载少量数据
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

## 项目结构
- `records/track_10min_16mb/` — 排行榜提交（每个提交一个文件夹）
- `records/track_non_record_16mb/` — 非纪录提交
- `data/` — 数据集和 tokenizer
- `logs/` — 训练日志输出

## 学习路径
用户是 PM 背景，通过分析排行榜提交来学习模型训练。
策略：分析他人方案 → 理解原理 → 复刻 → 迭代改进。

## 学习目标
通过这个挑战理解模型训练的核心概念：
- 什么是 loss、BPB、量化、attention、MLP
- 训练循环怎么运作：forward → loss → backward → update weights
- 模型压缩的权衡：参数数量 vs 参数精度 vs 模型质量
- 如何读懂别人的代码 diff 并复刻改进

## 已掌握的关键发现
1. 滑动窗口评估（~0.032 BPB）：不改模型，纯"考试技巧"，贡献最大
2. 更长序列（~0.019）：让模型看到更多上下文
3. MLP 3x 扩展（~0.015）：增大模型"思考"容量，靠更狠的量化塞进 16MB
4. XSA 注意力（~0.012）：减少注意力的"自恋"倾向，强制关注其他 token
5. Int6 QAT（~0.007）：训练时模拟量化噪声，消除压缩损失
6. FP16 embedding（~0.005）：embedding 对量化最敏感，保留高精度
7. SmearGate + BigramHash（~0.004）：免费注入相邻 token 的信息
8. EMA 权重平均（~0.004）：平滑权重分布，提升量化质量
9. LeakyReLU²（~0.003）：消除死神经元
10. 深度递归：整模型循环不如独立层（差 0.025）；但 4 月后只循环中间 2-3 层 + 并行残差成为主流栈
13. Score-first TTT（~0.002-0.003）：评估时先给一段打分、再用这段微调，边考边学
11. MoE 在 16MB 约束下不对症——省的是计算不是存储
12. 极端量化（1-bit/三值）可以塞更多参数但精度损失太大，目前不如主流栈

## 注意事项
- 本地 MLX 训练只用于快速验证想法，不用于正式提交
- M4 Max 上完整训练约 2.5 小时，评估约 12 分钟
- 正式提交需要在 RunPod 租 8×H100 跑 CUDA 版本
- 所有提交都是公开 PR，可以基于他人代码改进（注明 "On PR #xxx"）
