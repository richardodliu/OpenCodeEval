from loguru import logger

BENCHMARKS = [
    "HumanEval",
    "mbpp",
    "MBPP",
    "LeetCode",
    "BigCodeBench",
    "Bird",
    "Spider",
    "understandml"
]

SPLITS = {
    "HumanEval": ["base", "plus"],
    "mbpp": ["full", "sanitized"],
    "MBPP": ["base", "plus"],
    "LeetCode": ["contest", "train", "validation", "test"],
    "BigCodeBench": ["full", "hard"],
    "Bird": ["train", "dev"],
    "Spider": ["train", "dev"],
    "understandml": ["human", "model"]
}

def check_args(args):

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    # When eval the pretrained model, it can not response to the instrcution, so suggest using the Completion prompt
    if args.model_type == "Base" and args.prompt_type == "Instruction":
        logger.warning("Prompt type must be Completion for Base Model")

    # check the split is valid for the task
    if args.split not in SPLITS[args.task]:
        logger.error(f"split {args.split} is not valid for {args.task}, please use {SPLITS[args.task]}")

    # check the list_k is valid
    args.list_k = list(map(int, args.list_k.split(',')))
    if max(args.list_k) > args.num_samples:
        logger.warning("max of list_k must be less than num_samples")
        args.list_k = [k for k in args.list_k if k <= args.num_samples]

    # check the temperature is valid
    if args.num_samples > 1 and args.temperature == 0.0:
        logger.error("Temperature is not allowed when num_samples > 1")

    #When eval the chat model, it can not reply to the uncompleted code, so suggest using the Instruction prompt
    if args.model_type == "Chat" and args.prompt_type == "Completion":
        if args.prompt_prefix == "":
            logger.warning("Prompt prefix is not set, using default prefix")
            args.prompt_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n"
        if args.prompt_suffix == "":
            logger.warning("Prompt suffix is not set, using default suffix")
            args.prompt_suffix = "\n```\n"

    return args

def get_args(parser):

    #===================Model Parser===================
    parser.add_argument("--model_name", default = None, type=str)
    parser.add_argument("--tokenizer_name", default = None, type=str)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", default="vllm", type=str, choices=["openai", "vllm", "sglang", "transformer"])
    parser.add_argument("--task", default="HumanEval", type=str, choices = BENCHMARKS)
    parser.add_argument("--split", default="base", type = str)
    parser.add_argument("--prompt_type", default = "Instruction", type=str, choices=["Completion", "Instruction"])
    parser.add_argument("--model_type", default = "Chat", type = str, choices=["Base", "Chat"])
    parser.add_argument("--list_k", default = "1,3,5,10", type = str)
    parser.add_argument("--time_out", default = 3, type = float)
    
    #===================Computer Parser===================
    parser.add_argument("--num_gpus", default = 1, type=int)
    parser.add_argument("--num_workers", default = 1, type=int)

    parser.add_argument("--save_path", default = 'save', type=str)
    parser.add_argument("--batch_size", default = 164, type=int)
    parser.add_argument("--num_samples", default = 1, type=int)
    parser.add_argument("--max_tokens", default = 2048, type=int)
    parser.add_argument("--temperature", default = 0.0, type=float)
    parser.add_argument("--top_p", default = 1.0, type=float)

    #===================Prompt Parser===================
    parser.add_argument("--prompt_prefix", default = "", type=str)
    parser.add_argument("--prompt_suffix", default = "", type=str)
    parser.add_argument("--response_prefix", default = "", type=str)
    parser.add_argument("--response_suffix", default = "", type=str)

    return parser.parse_args()