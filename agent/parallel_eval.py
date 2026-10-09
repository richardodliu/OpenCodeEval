"""
并发评测脚本 - 并发评测目录下所有有效checkpoint

用法:
    python parallel_eval.py --checkpoint_dir /path/to/checkpoints \
                            --config_path config/leetcode.json \
                            --num_gpus 2 \
                            --num_workers 4 \
                            --max_parallel 4

说明:
    - 自动扫描checkpoint_dir下所有有效的checkpoint目录
    - 使用Ray并发执行评测任务
    - 每个checkpoint独立评测，互不干扰
    - 支持控制最大并行任务数
"""

import os
import re
import sys
import json
import subprocess
import ray
from loguru import logger
from collections import defaultdict
from argparse import ArgumentParser
from typing import Dict, List, Tuple

from lark_message import build_message, send_message
from jinja2 import Template


def load_json(json_path: str) -> dict:
    """加载JSON配置文件"""
    try:
        with open(json_path, 'r') as f:
            configs = json.load(f)
        return configs
    except Exception as e:
        logger.error(f"Error in load_json {json_path}. Error message: {e}")
        return {}


def eval_finish(save_path: str, benchmark_name: str) -> bool:
    """检查某个benchmark的评测是否已完成"""
    if benchmark_name not in os.listdir(save_path) or not os.path.isdir(os.path.join(save_path, benchmark_name)):
        logger.info(f"benchmark {benchmark_name} directory has not been created")
        return False
    if 'results.jsonl' not in os.listdir(os.path.join(save_path, benchmark_name)):
        logger.info(f"benchmark {benchmark_name} result has not been output")
        return False
    return True


def load_result(save_path: str, benchmark_name: str) -> float:
    """加载评测结果"""
    file_name = os.path.join(save_path, benchmark_name, 'results.jsonl')
    with open(file_name, 'r') as f:
        result = [json.loads(line.strip()) for line in f]
    return float(list(result[0].values())[0] * 100)


def is_valid_checkpoint(ckpt_path: str) -> bool:
    """
    判断是否是有效的checkpoint目录

    有效条件:
    1. 是一个目录
    2. 不以'eval'开头（排除评测结果目录）
    3. 包含模型文件（可选：可以根据实际情况添加更多条件）
    """
    if not os.path.isdir(ckpt_path):
        return False

    dirname = os.path.basename(ckpt_path)
    if dirname.startswith('eval'):
        return False

    # 检查是否包含模型文件: 例如检查是否有 config.json 或 pytorch_model.bin 等
    required_files = ['config.json']
    for f in required_files:
        if not os.path.exists(os.path.join(ckpt_path, f)):
            return False

    return True


def get_valid_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    递归获取目录下所有有效的checkpoint

    Args:
        checkpoint_dir: checkpoint根目录

    Returns:
        有效checkpoint的完整路径列表
    """
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        if is_valid_checkpoint(root):
            checkpoints.append(root)

    return checkpoints


def extract_number(path: str) -> int:
    """从路径中提取数字，用于排序"""
    basename = os.path.basename(path)
    match = re.search(r'\d+', basename)
    if match is None:
        return sys.maxsize
    return int(match.group())


@ray.remote
def evaluate_checkpoint(
    ckpt_path: str,
    benchmark_configs: Dict,
    num_gpus: int,
    num_workers: int
) -> Tuple[str, Dict]:
    """
    评测单个checkpoint（Ray remote函数）

    Args:
        ckpt_path: checkpoint完整路径
        benchmark_configs: benchmark配置字典
        num_gpus: 每个checkpoint使用的GPU数
        num_workers: 每个checkpoint使用的CPU worker数

    Returns:
        (checkpoint_name, results_dict)
    """
    # 在Ray worker中重新加载模板
    cmd_template = Template(open('config/cmd.jinja').read())

    ckpt_name = os.path.basename(ckpt_path)

    # 在checkpoint目录的父目录下创建eval结果目录
    parent_dir = os.path.dirname(ckpt_path)
    eval_dir = os.path.join(parent_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    save_path = os.path.join(eval_dir, ckpt_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results = {}

    logger.info(f"[{ckpt_name}] Starting evaluation with {len(benchmark_configs)} benchmarks")

    # 顺序执行该checkpoint的所有benchmark
    for benchmark, benchmark_config in benchmark_configs.items():
        if eval_finish(save_path, benchmark):
            logger.info(f"[{ckpt_name}][{benchmark}] Already finished, skipping")
            try:
                score = load_result(save_path, benchmark)
                results[benchmark] = float(score)
            except Exception as e:
                logger.exception(f"[{ckpt_name}][{benchmark}] Load result error: {e}")
            continue

        # 覆盖num_gpus和num_workers
        benchmark_config = {**benchmark_config, "num_gpus": num_gpus, "num_workers": num_workers}

        logger.info(f"[{ckpt_name}][{benchmark}] Starting evaluation")
        cmd = cmd_template.render(
            **{**benchmark_config, "model_name": ckpt_path,
               "save_path": os.path.join(save_path, benchmark)}
        )
        logger.info(f"[{ckpt_name}][{benchmark}] cmd: {cmd}")

        try:
            cmd_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if cmd_result.returncode != 0:
                logger.error(f"[{ckpt_name}][{benchmark}] Error: {cmd_result.stderr}")
            else:
                logger.info(f"[{ckpt_name}][{benchmark}] Completed successfully")
        except Exception as e:
            logger.exception(f"[{ckpt_name}][{benchmark}] Exception: {e}")

    # 收集所有结果
    for benchmark in benchmark_configs.keys():
        if benchmark in results:
            continue
        try:
            score = load_result(save_path, benchmark)
            results[benchmark] = float(score)
            logger.info(f"[{ckpt_name}][{benchmark}] result: {score}")
        except Exception as e:
            logger.exception(f"[{ckpt_name}][{benchmark}] get score error: {e}")

    logger.info(f"[{ckpt_name}] Evaluation completed with results: {results}")
    return ckpt_name, results


def parallel_evaluate(
    checkpoint_dir: str,
    benchmark_configs: Dict,
    num_gpus_per_ckpt: int,
    num_workers_per_ckpt: int,
    max_parallel: int = None,
    webhook_url: str = None,
    feishu_msg: bool = False
):
    """
    并发评测所有checkpoint

    Args:
        checkpoint_dir: checkpoint根目录
        benchmark_configs: benchmark配置
        num_gpus_per_ckpt: 每个checkpoint使用的GPU数
        num_workers_per_ckpt: 每个checkpoint使用的CPU worker数
        max_parallel: 最大并行任务数（None表示不限制）
        webhook_url: 飞书webhook地址
        feishu_msg: 是否发送飞书消息
    """
    # 初始化Ray
    if not ray.is_initialized():
        ray.init()

    # 获取可用GPU数量
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus == 0:
        logger.error("No GPUs available, exiting")
        return False
    logger.info(f"Available GPUs (from Ray): {total_gpus}")

    # 获取所有有效的checkpoint
    checkpoints = get_valid_checkpoints(checkpoint_dir)
    if not checkpoints:
        logger.error(f"No valid checkpoints found in {checkpoint_dir}")
        return False

    # 按照数字排序
    checkpoints = sorted(checkpoints, key=extract_number, reverse=True)
    logger.info(f"Found {len(checkpoints)} valid checkpoints: {[os.path.basename(c) for c in checkpoints]}")

    # 计算实际并行数
    if max_parallel is None:
        # 根据GPU数量自动计算
        max_parallel = max(1, total_gpus // num_gpus_per_ckpt)

    logger.info(f"Max parallel tasks: {max_parallel}")
    logger.info(f"Starting parallel evaluation: {len(checkpoints)} checkpoints, {num_gpus_per_ckpt} GPUs per ckpt")

    ckpt2result = defaultdict(dict)

    # 提交所有任务
    futures = [
        evaluate_checkpoint.options(num_gpus=num_gpus_per_ckpt).remote(
            ckpt_path, benchmark_configs, num_gpus_per_ckpt, num_workers_per_ckpt
        )
        for ckpt_path in checkpoints
    ]

    # 等待所有任务完成并收集结果
    logger.info("Waiting for all evaluation tasks to complete...")
    results = ray.get(futures)

    for ckpt_name, ckpt_results in results:
        ckpt2result[ckpt_name] = ckpt_results
        logger.info(f"[{ckpt_name}] Final results: {ckpt_results}")

    # 输出汇总结果
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    for ckpt_name in sorted(ckpt2result.keys(), key=extract_number, reverse=True):
        logger.info(f"{ckpt_name}:")
        for benchmark, score in ckpt2result[ckpt_name].items():
            logger.info(f"  {benchmark}: {score:.2f}")
    logger.info("=" * 80)

    # 发送飞书消息
    if feishu_msg and webhook_url:
        try:
            message = build_message(checkpoint_dir, ckpt2result)
            send_message(webhook_url, message)
            logger.info("Feishu message sent successfully")
        except Exception as e:
            logger.exception(f"Failed to send Feishu message: {e}")

    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="并发评测checkpoint目录下所有有效checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="checkpoint根目录，将评测该目录下所有有效的checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="benchmark配置文件路径")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="每个checkpoint使用的GPU数量")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="每个checkpoint使用的CPU worker数量")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="最大并行任务数（默认根据GPU数量自动计算）")
    parser.add_argument("--feishu_msg", action="store_true",
                        help="是否发送飞书通知")

    args = parser.parse_args()

    # 获取webhook_url
    webhook_url = os.environ.get('WEBHOOK_URL', None)

    try:
        # 加载benchmark配置
        benchmark_configs = load_json(args.config_path)
        if not benchmark_configs:
            logger.error(f"Failed to load benchmark configs from {args.config_path}")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("PARALLEL EVALUATION CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
        logger.info(f"Config path: {args.config_path}")
        logger.info(f"Benchmark configs: {list(benchmark_configs.keys())}")
        logger.info(f"GPUs per checkpoint: {args.num_gpus}")
        logger.info(f"Workers per checkpoint: {args.num_workers}")
        logger.info(f"Max parallel tasks: {args.max_parallel if args.max_parallel else 'auto'}")
        logger.info(f"Feishu notification: {args.feishu_msg}")
        logger.info("=" * 80)

        # 执行并发评测
        success = parallel_evaluate(
            checkpoint_dir=args.checkpoint_dir,
            benchmark_configs=benchmark_configs,
            num_gpus_per_ckpt=args.num_gpus,
            num_workers_per_ckpt=args.num_workers,
            max_parallel=args.max_parallel,
            webhook_url=webhook_url,
            feishu_msg=args.feishu_msg
        )

        if success:
            logger.info("Parallel evaluation completed successfully")
            sys.exit(0)
        else:
            logger.error("Parallel evaluation failed")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
