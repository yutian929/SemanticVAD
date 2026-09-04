import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tokenizers import AddedToken
from omegaconf import OmegaConf


def define_add_token():
    add_tokens = [AddedToken(f"<|audio_{x}|>") for x in range(51866)]

    return add_tokens


def define_special_token():
    specail_tokens = [
        "<|task_asr|>",
        "<|task_duplex_predict|>",
        "<|punctuation_on|>",
        "<|punctuation_off|>",
        "<|padding|>",
        "<|end_of_sentence|>",
        "<|begin_of_sentence|>",
        "<|user_complete|>",
        "<|user_backchannel|>",
        "<|user_incomplete|>",
        "<|assistant_backchannel|>",
        "<|user_idle|>",
        "<|user_nonidle|>",
        "<|assistant_interrupt|>",
        "<|state_tba05|>",
        "<|state_tba06|>",
        "<|state_tba07|>",
        "<|state_tba08|>",
        "<|state_tba09|>",
        "<|state_tba10|>",
        "<|state_tba11|>",
        "<|state_tba12|>",
        "<|state_tba13|>",
        "<|state_tba14|>",
        "<|state_tba15|>",
        "<|state_tba16|>",
        "<|state_tba17|>",
        "<|state_tba18|>",
        "<|state_tba19|>",
        "<|state_tba20|>",
        "<|state_tba21|>",
    ]

    return [AddedToken(x, special=True) for x in specail_tokens]


def parse_args():
    parser = argparse.ArgumentParser(description="Expand model vocabulary")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the original model"
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the expanded tokenizer",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the model with expanded vocabulary",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to load and save the model in FP16 precision",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to load and save the model in BF16 precision",
    )

    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to use safetensors to save the model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use, can be 'cpu', 'cuda', or 'auto'",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Determine data type for model loading
    torch_dtype = None
    if args.fp16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16

    # expand special token
    add_tokens = define_add_token()
    special_tokens = define_special_token()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    old_vocab_size = len(tokenizer)
    print(f"old vocab size: {old_vocab_size}")

    tokenizer.add_tokens(special_tokens, special_tokens=True)
    vocab_size_special = len(tokenizer)

    tokenizer.add_tokens(add_tokens)
    vocab_size_special_audio = len(tokenizer)
    print(f"new vocab size: {vocab_size_special_audio}")

    tokenizer.save_pretrained(args.tokenizer_path)
    print(f"save resized tokenizer to: {args.tokenizer_path}")

    vocab_size_to_expand = vocab_size_special
    print(f"Expanded vocabulary size (without audio token): {vocab_size_to_expand}")

    # vocab_size_to_expand = vocab_size_special_audio
    # print(f"Expanded vocabulary size (with audio token): {vocab_size_to_expand}")

    # Check if expansion is needed
    if vocab_size_to_expand <= old_vocab_size:
        print(
            "Expanded tokenizer vocabulary size is not larger than the original model vocabulary size, no expansion needed"
        )
        return

    # Load the original model configuration
    print(f"Loading original model configuration: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)

    # Load the original model
    print(f"Loading original model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype
    )

    # Get the current model vocabulary size
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Current model vocabulary size: {current_vocab_size}")

    if current_vocab_size < vocab_size_to_expand:
        # Expand the model vocabulary
        print(
            f"Expanding model vocabulary from {current_vocab_size} to {vocab_size_to_expand}"
        )
        model.resize_token_embeddings(vocab_size_to_expand)
    else:
        print("Model vocabulary size already matches tokenizer, no expansion needed")

    # init embedding_layer
    embedding_layer = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    # lm_head.weight.data = embedding_layer.weight.data.clone()
    with torch.no_grad():
        for i in range(vocab_size_to_expand - old_vocab_size):
            embedding_layer.weight[old_vocab_size + i].normal_(mean=0.0, std=0.02)
            lm_head.weight[old_vocab_size + i].normal_(mean=0.0, std=0.02)

    # Verify expansion
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"New model vocabulary size after expansion: {new_vocab_size}")

    # Update configuration
    config.vocab_size = new_vocab_size

    my_config_path = os.path.join(args.output_path, "config_record.yaml")
    my_config = OmegaConf.create({})
    my_config.lm_vocab_size = new_vocab_size
    my_config.tokenizer_vocab_size = vocab_size_special_audio
    my_config.bos_token_id = config.bos_token_id or tokenizer.bos_token_id
    my_config.eos_token_id = config.eos_token_id or tokenizer.eos_token_id
    my_config.pad_token_id = config.pad_token_id or tokenizer.pad_token_id
    my_config.added_token_start = old_vocab_size
    my_config.added_audio_token_start = vocab_size_special

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Save updated configuration
    config.save_pretrained(args.output_path)
    with open(my_config_path, "w") as f:
        OmegaConf.save(my_config, f)

    # Save the expanded model
    print(f"Saving model with expanded vocabulary to {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=args.safe_serialization)

    # Copy tokenizer to output directory (optional)
    if args.tokenizer_path != args.output_path:
        print(f"Copying tokenizer to output directory {args.output_path}")
        tokenizer.save_pretrained(args.output_path)

    print("Model vocabulary expansion completed!")


if __name__ == "__main__":
    main()

# python scripts/expand_vocab.py --model_path pretrained_models/Qwen3-1.7B --tokenizer_path pretrained_models/Qwen3-1.7B-expand_vocab --output_path pretrained_models/Qwen3-1.7B-expand_vocab --safe_serialization

# python -m debugpy --listen 5678 --wait-for-client scripts/expand_vocab.py --model_path pretrained_models/Qwen3-1.7B --tokenizer_path pretrained_models/Qwen3-1.7B-expand_vocab --output_path pretrained_models/Qwen3-1.7B-expand_vocab --safe_serialization
