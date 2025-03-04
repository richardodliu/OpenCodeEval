# OpenCodeEval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![vLLM](https://img.shields.io/badge/vLLM-supported-green.svg)](https://github.com/vllm-project/vllm) [![OpenAI](https://img.shields.io/badge/OpenAI-compatible-brightgreen.svg)](https://openai.com/) [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Downloads](https://img.shields.io/github/downloads/yourusername/OpenCodeEval/total.svg)](https://github.com/yourusername/OpenCodeEval/releases) [![Stars](https://img.shields.io/github/stars/yourusername/OpenCodeEval.svg)](https://github.com/yourusername/OpenCodeEval/stargazers) [![Issues](https://img.shields.io/github/issues/yourusername/OpenCodeEval.svg)](https://github.com/yourusername/OpenCodeEval/issues)

OpenCodeEval is a comprehensive framework for evaluating Large Language Models (LLMs) on code generation tasks. It provides standardized benchmarks, flexible configurations, and robust evaluation metrics to assess model performance across different programming challenges.

## Overview

OpenCodeEval is a robust framework designed to evaluate LLMs' performance on code generation tasks. It supports multiple benchmark datasets and provides flexible evaluation configurations.

## Features

- Multiple benchmark dataset support:
  - HumanEval & HumanEvalPlus
  - MBPP & MBPPPlus
  - BigCodeBench & BigCodeBench-Hard
  - LeetCode

- Flexible model support:
  - Base models
  - Chat models

- Backend support:
  - vLLM acceleration
  - Sglang acceleration
  - OpenAI API integration

- Comprehensive evaluation tools:
  - Pass@k metrics
  - Multiple sample evaluation
  - Parallel processing

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/richardodliu/OpenCodeEval.git
cd OpenCodeEval
```

2. Download benchmark datasets:

```bash
cd src/data
bash dataset.sh
```

3. Install dependencies:

```bash
pip install -e .
```

4. Run evaluation:

Basic usage:
```bash
OpenCodeEval  --model_name <your_model_name> \
              --save_path <output_directory> \
              --num_gpus <number_of_gpus> \
              --batch_size <batch_size> \
              --task <benchmark_name>
```

Complete example:
```bash
OpenCodeEval  --model_name '/path/to/your/model/checkpoint' \
              --task 'LeetCodeTest' \
              --save 'test/output' \
              --num_gpus 1 \
              --num_samples 1 \
              --list_k '1' \
              --temperature 0.0 \
              --num_workers 10 \
              --batch_size 200 \
              --max_tokens 4096 \
              --model_type 'Chat' \
              --prompt_type 'Instruction' \
              --prompt_prefix '' \
              --prompt_suffix '' \
              --trust_remote_code
```

## Supported Benchmarks

### 1. HumanEval
- Standard code generation benchmark
- Function completion tasks
- Python programming problems
- Automated test cases

### 2. MBPP (Mostly Basic Programming Problems)
- Basic programming tasks
- Few-shot learning support
- Python implementation
- Test-driven evaluation

### 3. BigCodeBench
- Comprehensive coding tasks
- Multiple difficulty levels
- Various programming challenges
- Extensive test coverage

### 4. LeetCode
- Algorithm problems
- Data structure challenges
- Multiple difficulty levels
- Real-world coding scenarios

## Project Structure
```
OpenCodeEval/
├── src/
│   ├── backend/         # Model backend implementations
│   ├── benchmark/       # Benchmark dataset implementations
│   ├── data/           # Dataset files
│   ├── eval/           # Evaluation utilities
│   └── main.py         # Main entry point
├── LICENSE             # Apache 2.0 license
└── README.md
```

## Configuration

The framework supports various configuration options:

- Model configurations:
  - Model type (Base/Chat)
  - Number of GPUs
  - Batch size
  - Temperature
  - Max tokens

- Prompt configurations:
  - Prompt type (Completion/Instruction)
  - Prompt prefix/suffix
  - Stop words

- Evaluation configurations:
  - Number of samples
  - Number of workers
  - Timeout settings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation

If you use OpenCodeEval in your research, please cite:

```bibtex
@software{OpenCodeEval,
  title = {OpenCodeEval: An Extensible, Efficient, and Easy-to-use Evaluation Framework for Code Generation Tasks on Large Language Models},
  author = {Ren-Biao Liu, Yun-Hui Xia, Wei Shen, Tian-Hao Cheng, Chong-Han Liu},
  year = {2024},
  url = {https://github.com/richardodliu/OpenCodeEval}
}
```

## Acknowledgments

We would like to thank the following projects and individuals for their contributions to OpenCodeEval:

### Datasets
- [HumanEval](https://github.com/openai/humaneval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- [EvalPlus](https://github.com/evalplus/evalplus)
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
- [LeetCode](https://github.com/newfacade/LeetCodeDataset/)

### Backends
- [vLLM](https://github.com/vllm-project/vllm)
- [Sglang](https://github.com/sgl-project/sglang)
- [OpenAI](https://github.com/openai/openai-python)

## Contact

For questions and feedback, please open an issue in the GitHub repository.