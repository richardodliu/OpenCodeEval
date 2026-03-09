# Parallel Evaluation Script (parallel_eval.py)

## 概述

`parallel_eval.py` 是基于 `main.py` 创建的并发评测脚本，专门用于并发评测指定目录下所有有效的checkpoint。

## 与原版本 (main.py) 的主要区别

| 特性 | main.py | parallel_eval.py |
|------|---------|------------------|
| **输入方式** | 指定一个checkpoint根目录，扫描子目录 | 指定一个目录，直接评测该目录下所有checkpoint |
| **checkpoint查找** | 查找 `checkpoint_path` 下的子目录（如 iter_0001, iter_0002） | 直接查找 `checkpoint_dir` 下的所有有效checkpoint目录 |
| **使用场景** | 适合训练中持续产生checkpoint的场景 | 适合一次性并发评测多个独立checkpoint |
| **结果保存** | 保存在 `checkpoint_path/eval/` 下 | 保存在 `checkpoint_dir/eval/` 下 |
| **并发控制** | 基于可用GPU自动并发 | 支持 `--max_parallel` 参数控制并发数 |

## 功能特性

1. **自动checkpoint发现**: 自动扫描目录下所有有效的checkpoint
2. **并发评测**: 使用Ray框架实现多checkpoint并发评测
3. **智能跳过**: 自动跳过已完成的评测任务
4. **结果汇总**: 评测完成后输出汇总报告
5. **飞书通知**: 支持评测完成后发送飞书通知
6. **灵活配置**: 支持自定义GPU数、worker数、并行数等参数

## 有效checkpoint判定规则

一个目录被视为有效checkpoint需满足以下条件:

1. 是一个目录（非文件）
2. 目录名不以 `eval` 开头（排除评测结果目录）
3. 可选: 包含必要的模型文件（可根据需要在代码中自定义）

## 使用方法

### 基本用法

```bash
python parallel_eval.py \
    --checkpoint_dir /path/to/checkpoints \
    --config_path config/leetcode.json \
    --num_gpus 2 \
    --num_workers 4
```

### 完整参数说明

```bash
python parallel_eval.py \
    --checkpoint_dir <checkpoint目录> \     # 必需: checkpoint根目录
    --config_path <配置文件路径> \          # 必需: benchmark配置文件
    --num_gpus <GPU数量> \                  # 可选: 每个checkpoint使用的GPU数，默认1
    --num_workers <worker数量> \            # 可选: 每个checkpoint使用的CPU worker数，默认4
    --max_parallel <最大并行数> \           # 可选: 最大并行任务数，默认根据GPU数自动计算
    --feishu_msg                            # 可选: 是否发送飞书通知
```

### 参数详解

- `--checkpoint_dir`: checkpoint根目录，脚本会扫描该目录下所有有效的checkpoint子目录
- `--config_path`: benchmark配置文件路径（JSON格式），定义要运行的benchmark任务
- `--num_gpus`: 每个checkpoint评测使用的GPU数量
- `--num_workers`: 每个checkpoint评测使用的CPU worker数量
- `--max_parallel`: 限制最大并行任务数
  - 如果不指定，会根据可用GPU数自动计算: `total_gpus // num_gpus`
  - 例如: 8个GPU，每个任务用2个GPU，则最多并行4个任务
- `--feishu_msg`: 启用飞书通知（需要设置环境变量 `WEBHOOK_URL`）

## 使用示例

### 示例1: 基本评测

```bash
# 评测 /data/checkpoints 目录下所有checkpoint
# 每个checkpoint使用1个GPU，4个worker
python parallel_eval.py \
    --checkpoint_dir /data/checkpoints \
    --config_path config/leetcode.json \
    --num_gpus 1 \
    --num_workers 4
```

### 示例2: 限制并行数

```bash
# 最多同时评测2个checkpoint
python parallel_eval.py \
    --checkpoint_dir /data/checkpoints \
    --config_path config/leetcode.json \
    --num_gpus 2 \
    --num_workers 4 \
    --max_parallel 2
```

### 示例3: 带飞书通知

```bash
# 设置飞书webhook
export WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"

# 运行评测并发送通知
python parallel_eval.py \
    --checkpoint_dir /data/checkpoints \
    --config_path config/leetcode.json \
    --num_gpus 2 \
    --num_workers 4 \
    --feishu_msg
```

## 目录结构示例

### 输入目录结构
```
/data/checkpoints/
├── model_v1/          # checkpoint 1
│   ├── config.json
│   └── pytorch_model.bin
├── model_v2/          # checkpoint 2
│   ├── config.json
│   └── pytorch_model.bin
├── model_v3/          # checkpoint 3
│   ├── config.json
│   └── pytorch_model.bin
└── eval/              # (会被自动跳过)
```

### 输出目录结构
```
/data/checkpoints/
├── model_v1/
├── model_v2/
├── model_v3/
└── eval/              # 评测结果目录
    ├── model_v1/
    │   ├── leetcode_train/
    │   │   └── results.jsonl
    │   └── leetcode_test/
    │       └── results.jsonl
    ├── model_v2/
    │   └── ...
    └── model_v3/
        └── ...
```

## 并发机制说明

1. **GPU资源分配**: 每个checkpoint评测任务独占 `--num_gpus` 个GPU
2. **自动并发控制**: Ray会根据可用GPU数量自动调度任务
3. **任务隔离**: 每个checkpoint的评测是独立的，互不干扰
4. **失败处理**: 某个checkpoint评测失败不影响其他checkpoint

## 配置文件格式

benchmark配置文件格式与 `main.py` 相同，参考 `config/leetcode.json`:

```json
{
    "leetcode_train": {
        "task": "LeetCode",
        "split": "train",
        "backend": "vllm",
        "batch_size": 2641,
        "temperature": 0.0,
        "num_samples": 1,
        "list_k": "1",
        "max_tokens": 1024,
        "time_out": 3,
        "prompt_type": "Instruction",
        "model_type": "Chat",
        "prompt_prefix": "",
        "prompt_suffix": ""
    },
    "leetcode_test": {
        ...
    }
}
```

## 日志输出

脚本会输出详细的日志信息，包括:

- 发现的checkpoint列表
- 每个checkpoint的评测进度
- 每个benchmark的执行状态
- 评测结果汇总

示例输出:
```
================================================================================
BATCH EVALUATION CONFIGURATION
================================================================================
Checkpoint directory: /data/checkpoints
Config path: config/leetcode.json
Benchmark configs: ['leetcode_train', 'leetcode_test']
GPUs per checkpoint: 2
Workers per checkpoint: 4
Max parallel tasks: auto
================================================================================
Found 3 valid checkpoints: ['model_v3', 'model_v2', 'model_v1']
Max parallel tasks: 4
Starting batch evaluation: 3 checkpoints, 2 GPUs per ckpt
...
================================================================================
EVALUATION SUMMARY
================================================================================
model_v3:
  leetcode_train: 85.32
  leetcode_test: 72.45
model_v2:
  leetcode_train: 83.21
  leetcode_test: 70.12
model_v1:
  leetcode_train: 80.15
  leetcode_test: 68.90
================================================================================
```

## 注意事项

1. **GPU资源**: 确保有足够的GPU资源，否则任务会排队等待
2. **磁盘空间**: 确保有足够的磁盘空间存储评测结果
3. **Ray初始化**: 脚本会自动初始化Ray，无需手动启动Ray集群
4. **checkpoint验证**: 可以根据实际需求修改 `is_valid_checkpoint()` 函数来自定义checkpoint验证规则
5. **结果覆盖**: 已完成的评测不会被重复执行，如需重新评测需手动删除结果目录

## 依赖要求

- Python >= 3.7
- ray
- loguru
- jinja2
- 其他依赖与 `main.py` 相同

## 故障排查

### 问题1: 找不到有效checkpoint
- 检查 `--checkpoint_dir` 路径是否正确
- 检查目录下是否包含有效的checkpoint子目录
- 修改 `is_valid_checkpoint()` 函数的验证逻辑

### 问题2: GPU资源不足
- 减少 `--num_gpus` 参数
- 使用 `--max_parallel` 限制并发数
- 检查是否有其他程序占用GPU

### 问题3: 评测结果未生成
- 检查日志中的错误信息
- 确认benchmark配置正确
- 检查模型路径和权限

## 与原版本兼容性

- 配置文件格式完全兼容
- 评测结果格式完全兼容
- 可以与 `main.py` 共享同一套配置文件和工具函数

## 开发者信息

基于 `main.py` 修改，主要改动:

1. 重构checkpoint发现逻辑: `get_valid_checkpoints()`
2. 简化Ray任务函数: `evaluate_checkpoint()`
3. 增强并发控制: `--max_parallel` 参数
4. 优化日志输出: 评测汇总报告
