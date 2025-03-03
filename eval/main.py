import os
import re
import sys
import json
import subprocess
from loguru import logger
from collections import defaultdict
from argparse import ArgumentParser

from lark_message import build_message, send_message
from jinja2 import Template

CMD_TEMPLATE = Template(open('config/cmd.jinja').read())

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
    if 'result.jsonl' not in os.listdir(os.path.join(save_path, benchmark_name)):
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

def eval_loop(ckpt_path, benchmark_configs, webhook_url, feishu_msg):

    ckpt2result = defaultdict(dict)

    if not os.path.exists(ckpt_path):
        logger.warning(f'{ckpt_path} has not been created')
        return False

    steps = get_ckpt(ckpt_path)
    steps = sorted(steps, key = extract_number, reverse=True)

    logger.info(f"all step: {steps}")

    if not os.path.exists(os.path.join(ckpt_path, 'eval')):
        os.makedirs(os.path.join(ckpt_path, 'eval'))
    save_path = os.path.join(ckpt_path, 'eval')

    for step in steps:

        step_ckpt_path = os.path.join(ckpt_path, step)

        step_save_path = os.path.join(save_path, step)
        if not os.path.exists(step_save_path):
            os.makedirs(step_save_path)
        logger.info(f"eval save path for this step: {step_save_path}")

        for benchmark, benchmark_config in benchmark_configs.items():

            if not eval_finish(step_save_path, benchmark):
                logger.info(f"benchmark {benchmark} has not been finished")
                cmd = CMD_TEMPLATE.render(
                    **{**benchmark_config, "model_name": step_ckpt_path, "save_path": os.path.join(step_save_path, benchmark)}
                )   
                logger.info(f"cmd: {cmd}")

                try:
                    cmd_result = subprocess.run(cmd, shell=True)
                except Exception as e: 
                    logger.exception(f"Error in cmd {cmd}. Error message: {e}")
                    continue

                if cmd_result.returncode != 0:
                    logger.error(f"Error in handler {benchmark}. Error message: {cmd_result.stderr}")
                    continue

            try:
                score = load_result(step_save_path, benchmark)
                ckpt2result[step][benchmark] = float(score)
                logger.info(f"Step {step} on {benchmark} result exist, result: {score}")
            except Exception as e:
                logger.exception(f"get score error, step: {step}, handler {benchmark}\ndetails: {e}")
                logger.info(f"retrying handler {benchmark} in step {step}")

    if feishu_msg:
        message = build_message(ckpt_path, ckpt2result)
        send_message(webhook_url, message)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--feishu_msg", action="store_true")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path
    feishu_msg = bool(args.feishu_msg)


    try:
        webhook_url = os.environ['WEBHOOK_URL']
    except Exception as e:
        logger.error(f'webhook_url has not been configured, exception: {e}')
        sys.exit(1)

    try:
        benchmark_configs = load_json(config_path)
        logger.info(f"benchmark_configs: {benchmark_configs}")
        logger.info(f"checkpoint_path: {checkpoint_path}")
        eval_loop(checkpoint_path, benchmark_configs, webhook_url, feishu_msg)
    except Exception as e:
        logger.exception(f"[INFO] error: {e}")
        message = f"Error: {e}"
        sys.exit(0)
