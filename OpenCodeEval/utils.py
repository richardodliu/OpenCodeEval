import os
import re
import gzip
import json
import itertools
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Union, Iterable, Callable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

def refine_text(text: str, add_new_line: bool = True) -> str:
    text =  text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if add_new_line:
        return text.strip() + "\n"
    else:
        return text.strip()

def multi_process_function(function: Callable,
                           parameters: List,
                           num_workers: int = 1,
                           desc: str = "Completing tasks"):
    
    if num_workers > len(parameters) or num_workers > os.cpu_count():
        num_workers = min(os.cpu_count(), len(parameters))

    with ThreadPoolExecutor(num_workers) as executor:
        futures = []
        for param in parameters:
            future = executor.submit(function, param)
            futures.append(future)
            
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            results.append(result)

    return results

def markdown_extract(text: str, mode: Literal["first", "last", "all"] = "all") -> str:
    """
    Extract content enclosed by triple backticks (```) in a Markdown text.

    Args:
        text (str): The Markdown text to extract from.
        mode (Literal["first", "last", "all"]):
            - "first": Extract the first block of content.
            - "last": Extract the last block of content.
            - "all": Extract all blocks of content and join them with double newlines.

    Returns:
        str: Extracted content based on the specified mode.
    """
    # Match content inside triple backticks, ignoring the ``` lines
    pattern = r"```[ \t]*[\r\n]+[ \t]*(.*?)[ \t]*[\r\n]+[ \t]*```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        if mode == "first":
            return matches[0]
        elif mode == "last":
            return matches[-1]
        elif mode == "all":
            return "\n\n".join(matches)
    else:
            return ""

def program_extract(
    text: str,
    program: str = "python",
    mode: Literal["first", "last", "all"] = "all"
) -> str:

    program_pattern = rf"```{program}[ \t]*[\r\n]+[ \t]*(.*?)[ \t]*[\r\n]+[ \t]*```"
    program_re = re.compile(program_pattern, re.DOTALL | re.IGNORECASE)

    matches = program_re.findall(text)
    if matches:
        if mode == "first":
            return matches[0]
        elif mode == "last":
            return matches[-1]
        elif mode == "all":
            return "\n\n".join(matches)
    else:
        return ""

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def calculate_pass_at_k(
    evaluations: List[Dict], 
    num_samples: int, 
    list_k: List[int]
    ) -> List[Dict]:

    task_results = defaultdict(int)
    for evaluation in evaluations:
        task_results[evaluation['task_id']] += evaluation['passed']
    task_results = list(task_results.values())
    benchmark_results = []
    for k in list_k:
        pass_rate = float(np.mean(estimate_pass_at_k(num_samples = num_samples, num_correct = task_results, k = k)))
        benchmark_results.append({f"Pass@{k}": pass_rate})
    return benchmark_results