import argparse
import time
import mlx.core as mx
import numpy as np

# Import necessary functions and classes from mlx_bitnet
from mlx_bitnet import load_causal_model, BitnetTokenizer

def main(args):
    """
    Main function to load the model and perform text generation.
    """
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the model and tokenizer using the provided function
    # This function expects the original model name and will look
    # for the corresponding .npz file (e.g., "1bitLLM-bitnet_b1_58-xl.npz")
    print(f"[-] Loading model '{args.model}'...")
    model, tokenizer = load_causal_model(args.model, dtype=args.dtype)
    print(f"[+] Model loaded successfully.")
    print(f"    - Model Type: {type(model).__name__}")
    print(f"    - Tokenizer Type: {type(tokenizer).__name__}")
    print(f"    - Vocab Size: {tokenizer.vocab_size}")
    print(f"    - EOS token id: {tokenizer.eos_token_id}")

    # Tokenize the input prompt
    # Using return_tensors="pt" as tokenizer likely defaults to PyTorch tensors
    # We then convert them to numpy and then to mlx arrays
    print("[-] Tokenizing prompt...")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = mx.array(inputs.input_ids.numpy())
    attention_mask = mx.array(inputs.attention_mask.numpy())
    print("[+] Prompt tokenized.")
    print(f"    - Input IDs shape: {input_ids.shape}")

    print("[-] Starting generation...")
    print(f"Prompt: {args.prompt}", end="", flush=True)

    tokens = []
    skip = 0
    prompt_processing_finished = False
    start_time = time.perf_counter()

    # Use the model's generate method (which is a generator)
    # Note: The current generate implementation in mlx_bitnet.py doesn't have an EOS check
    # We'll add one here based on max_tokens and tokenizer.eos_token_id
    for token in model.generate(input_ids, attention_mask, temp=args.temp):
        if not prompt_processing_finished:
            # Evaluate the first token to measure prompt processing time
            mx.eval(token)
            prompt_processing_time = time.perf_counter() - start_time
            prompt_processing_finished = True

        tokens.append(token)

        # Check for End Of Sentence token
        if token.item() == tokenizer.eos_token_id:
             print("\n[+] EOS token detected.")
             break

        # Stop if max tokens generated
        if len(tokens) >= args.max_tokens:
            print(f"\n[+] Reached max tokens ({args.max_tokens}).")
            break

        # Decode and print incrementally
        # `mx.eval` is needed for the computation to happen and be decoded
        mx.eval(token) # Evaluate just the latest token for efficiency
        current_token_text = tokenizer.decode([token.item()])
        print(current_token_text, end="", flush=True)


    # Final evaluation of all tokens (might be redundant if evaluated incrementally)
    mx.eval(tokens)
    generation_end_time = time.perf_counter()
    generation_time = generation_end_time - start_time

    # Decode the final generated sequence
    full_generation = tokenizer.decode([t.item() for t in tokens])
    # The full text is already printed incrementally, but we can print it again if needed
    # print(f"\nComplete Generation: {args.prompt}{full_generation}")
    print("\n[-] Generation finished.")

    # Calculate and print performance metrics
    num_generated_tokens = len(tokens)
    generation_only_time = generation_time - prompt_processing_time
    tokens_per_sec = num_generated_tokens / generation_only_time if generation_only_time > 0 else 0

    print("\n[Performance Metrics]")
    print(f"  - Prompt processing time: {prompt_processing_time:.3f} s")
    print(f"  - Generation time (excl. prompt): {generation_only_time:.3f} s")
    print(f"  - Total generation time: {generation_time:.3f} s")
    print(f"  - Number of tokens generated: {num_generated_tokens}")
    print(f"  - Tokens per second: {tokens_per_sec:.3f} tok/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with MLX Bitnet model.")
    parser.add_argument(
        "--model",
        type=str,
        default="1bitLLM/bitnet_b1_58-xl",
        help="The Hugging Face name of the Bitnet model. The script will look for a corresponding .npz file (e.g., '1bitLLM-bitnet_b1_58-xl.npz').",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="The input prompt for text generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="The sampling temperature for generation (0 for greedy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation.",
    )
    parser.add_argument(
        "--dtype",
        help="The model data type for loading weights.",
        type=str,
        choices=["float16", "float32"],
        default="float16", # Usually float16 is preferred for inference
    )

    args = parser.parse_args()
    main(args)
