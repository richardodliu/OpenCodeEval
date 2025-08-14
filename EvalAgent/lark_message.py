import json
import time, requests, re
from loguru import logger
from collections import defaultdict

def post_with_retry(url, headers, data, max_num=3):
    cur_num = 0
    while cur_num < max_num:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.text
        except Exception as e:
            cur_num += 1
            logger.debug(f'connect to feishu failed: {str(e)}, retried times: {cur_num}')
            if cur_num == max_num:
                logger.error(f'connect to feishu failed: {str(e)}, raise error')
                return None
            time.sleep(5)



def build_message(ckpt_name, job_result):
    exp_name = '+'.join(ckpt_name.split('/')[-4:])
    message = f'====={exp_name}=====\n\n'
    handler_result = defaultdict(lambda: defaultdict(lambda: str()))

    for step, results in job_result.items():
        for handler,score in results.items():
            handler_result[handler][step] = score
    
    for handler, step2result in handler_result.items():
        message += f"[{handler}]\n"

        for step, result in step2result.items():
            # num = step.split('-')[-1]
            match = re.search(r'\d+', step)
            if match:
                num = int(match.group())
            else:
                num = step
            result = str(result).strip()[:6]
            message += f"step: {num}, result: {result}\n"
        
        message += '\n\n'
    
    return message



def send_message(url, message):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "msg_type": "text",
        "content": {
            "text": message
        }
    }
    json_data = json.dumps(data)
    num = 0
    response = ""
    while response.strip() != '<Response [200]>':
        response = str(requests.post(url, headers=headers, data=json_data))
        if response.strip() == '<Response [200]>':
            break
        num = num + 1
        time.sleep(3)
        if num == 5:
            return