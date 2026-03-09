import os
import re
import sys
import json
import subprocess
import ray
from loguru import logger
from collections import defaultdict
from argparse import ArgumentParser

from lark_message import build_message, send_message
from jinja2 import Template


def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            configs = json.load(f)
        return configs
    except Exception as e:
        logger.error(f"Error in load_json {json_path}. Error message: {e}")
        return {}

def eval_finish(save_path, benchmark_name):
    if benchmark_name not in os.listdir(save_path) or not os.path.isdir(os.path.join(save_path, benchmark_name)):
        logger.info(f"benchmark {benchmark_name} directory has not been created")
        return False
    if 'results.jsonl' not in os.listdir(os.path.join(save_path, benchmark_name)):
        logger.info(f"benchmark {benchmark_name} result has not been output")
        return False

    return True

def load_result(save_path, benchmark_name):
    file_name = os.path.join(save_path, benchmark_name, 'results.jsonl')
    with open(file_name, 'r') as f:
        result = [json.loads(line.strip()) for line in f]

    return float(list(result[0].values())[0] * 100)

def get_ckpt(ckpt_path):
    return [step for step in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, step)) and not step.startswith('eval')]

def extract_number(step):
    match = re.search(r'\d+', step)
    if match is None:
        return sys.maxsize
    return int(match.group())


@ray.remote
def run_step_evaluation(step, ckpt_path, save_path, benchmark_configs, num_gpus, num_workers):
    """执行单个 checkpoint 的所有 benchmark 评估（Ray remote 函数）"""
    # 在 Ray worker 中重新加载模板
    cmd_template = Template(open('config/cmd.jinja').read())

    step_ckpt_path = os.path.join(ckpt_path, step)
    step_save_path = os.path.join(save_path, step)

    if not os.path.exists(step_save_path):
        os.makedirs(step_save_path)

    results = {}

    # 顺序执行该 checkpoint 的所有 benchmark
    for benchmark, benchmark_config in benchmark_configs.items():
        if eval_finish(step_save_path, benchmark):
            logger.info(f"[{step}][{benchmark}] Already finished, skipping")
            continue

        # 覆盖 num_gpus 和 num_workers
        benchmark_config = {**benchmark_config, "num_gpus": num_gpus, "num_workers": num_workers}

        logger.info(f"[{step}][{benchmark}] Starting evaluation")
        cmd = cmd_template.render(
            **{**benchmark_config, "model_name": step_ckpt_path,
               "save_path": os.path.join(step_save_path, benchmark)}
        )
        logger.info(f"[{step}][{benchmark}] cmd: {cmd}")

        try:
            cmd_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if cmd_result.returncode != 0:
                logger.error(f"[{step}][{benchmark}] Error: {cmd_result.stderr}")
        except Exception as e:
            logger.exception(f"[{step}][{benchmark}] Exception: {e}")

    # 收集结果
    for benchmark in benchmark_configs.keys():
        try:
            score = load_result(step_save_path, benchmark)
            results[benchmark] = float(score)
            logger.info(f"[{step}][{benchmark}] result: {score}")
        except Exception as e:
            logger.exception(f"[{step}][{benchmark}] get score error: {e}")

    return step, results

def eval_loop(ckpt_path, benchmark_configs, webhook_url, feishu_msg, num_gpus_per_ckpt, num_workers_per_ckpt):

    ckpt2result = defaultdict(dict)

    if not os.path.exists(ckpt_path):
        logger.warning(f'{ckpt_path} has not been created')
        return False

    # 初始化 Ray（自动检测 GPU）
    if not ray.is_initialized():
        ray.init()

    # 从 Ray 获取可用 GPU 数量
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus == 0:
        logger.error("No GPUs available, exiting")
        return False
    logger.info(f"Available GPUs (from Ray): {total_gpus}")

    steps = get_ckpt(ckpt_path)
    steps = sorted(steps, key=extract_number, reverse=True)

    logger.info(f"all steps: {steps}")

    if not os.path.exists(os.path.join(ckpt_path, 'eval')):
        os.makedirs(os.path.join(ckpt_path, 'eval'))
    save_path = os.path.join(ckpt_path, 'eval')

    # 使用 Ray 并行执行
    logger.info(f"Starting parallel evaluation with {len(steps)} checkpoints, {num_gpus_per_ckpt} GPUs per ckpt")

    # 提交所有任务，动态指定资源
    futures = [
        run_step_evaluation.options(num_gpus=num_gpus_per_ckpt).remote(
            step, ckpt_path, save_path, benchmark_configs, num_gpus_per_ckpt, num_workers_per_ckpt
        )
        for step in steps
    ]

    # 等待所有任务完成并收集结果
    results = ray.get(futures)
    for step_name, step_results in results:
        ckpt2result[step_name] = step_results
        logger.info(f"[{step_name}] Completed with results: {step_results}")

    if feishu_msg:
        message = build_message(ckpt_path, ckpt2result)
        send_message(webhook_url, message)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per checkpoint")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers per checkpoint")
    parser.add_argument("--feishu_msg", action="store_true")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path
    num_gpus = args.num_gpus
    num_workers = args.num_workers
    feishu_msg = bool(args.feishu_msg)


    try:
        webhook_url = os.environ['WEBHOOK_URL']
    except Exception as e:
        webhook_url = None

    try:
        benchmark_configs = load_json(config_path)
        logger.info(f"benchmark_configs: {benchmark_configs}")
        logger.info(f"checkpoint_path: {checkpoint_path}")
        logger.info(f"num_gpus per ckpt: {num_gpus}, num_workers per ckpt: {num_workers}")
        eval_loop(checkpoint_path, benchmark_configs, webhook_url, feishu_msg, num_gpus, num_workers)
    except Exception as e:
        logger.exception(f"[INFO] error: {e}")
        message = f"Error: {e}"
        sys.exit(0)
