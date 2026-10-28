"""Smoke test: load the local GGUF model with llama.cpp and answer one question.

No retrieval happens here. This only checks that the model loads, uses the
Metal GPU, and generates text, and it reports load time and tokens/second.
"""

import argparse
import sys
import time
from pathlib import Path

from llama_cpp import Llama, llama_supports_gpu_offload

DEFAULT_MODEL = Path("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
DEFAULT_QUESTION = "What are the common symptoms of Type 2 Diabetes?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    print(f"GPU offload supported: {llama_supports_gpu_offload()}")

    start = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    load_seconds = time.perf_counter() - start
    print(f"Model loaded in {load_seconds:.1f}s")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical information assistant. Answer briefly and "
                "remind the user to consult a healthcare professional."
            ),
        },
        {"role": "user", "content": args.question},
    ]

    start = time.perf_counter()
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    gen_seconds = time.perf_counter() - start

    print("\n--- Answer ---")
    print(result["choices"][0]["message"]["content"].strip())

    usage = result["usage"]
    tokens = usage["completion_tokens"]
    print("\n--- Stats ---")
    print(f"Prompt tokens:     {usage['prompt_tokens']}")
    print(f"Generated tokens:  {tokens}")
    print(f"Generation time:   {gen_seconds:.1f}s")
    print(f"Speed:             {tokens / gen_seconds:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
