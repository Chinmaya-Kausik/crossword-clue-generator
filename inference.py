"""
Inference script for the fine-tuned Qwen 3 0.6B LoRA model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


PROMPT_TEMPLATE = """Generate a cryptic crossword clue for the following:
Answer: {answer}
Enumeration: {enumeration}

Clue:"""


def load_model(base_model_name: str, lora_path: str, device: str = "mps"):
    """Load the base model with LoRA weights."""
    print(f"Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights from {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    # Set device
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_clue(
    model,
    tokenizer,
    answer: str,
    enumeration: str,
    device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
):
    """Generate cryptic crossword clue(s) for a given answer."""
    prompt = PROMPT_TEMPLATE.format(answer=answer, enumeration=enumeration)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    clues = []
    for output in outputs:
        generated = tokenizer.decode(output, skip_special_tokens=True)
        clue = generated[len(prompt):].strip()
        clues.append(clue)

    return clues if num_return_sequences > 1 else clues[0]


def interactive_mode(model, tokenizer, device):
    """Run interactive clue generation."""
    print("\n" + "=" * 50)
    print("Cryptic Crossword Clue Generator")
    print("=" * 50)
    print("Enter 'quit' to exit\n")

    while True:
        answer = input("Answer: ").strip().upper()
        if answer.lower() == "quit":
            break

        enumeration = input("Enumeration (e.g., '6' or '3,4'): ").strip()
        if enumeration.lower() == "quit":
            break

        num_clues = input("Number of clues to generate (default 3): ").strip()
        num_clues = int(num_clues) if num_clues else 3

        print("\nGenerating clues...")
        clues = generate_clue(
            model, tokenizer, answer, enumeration, device,
            num_return_sequences=num_clues,
        )

        print(f"\nGenerated clues for '{answer}' ({enumeration}):")
        if isinstance(clues, list):
            for i, clue in enumerate(clues, 1):
                print(f"  {i}. {clue}")
        else:
            print(f"  {clues}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cryptic crossword clues")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/best",
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
    )
    parser.add_argument(
        "--answer",
        type=str,
        help="Answer to generate clue for (skip for interactive mode)",
    )
    parser.add_argument(
        "--enumeration",
        type=str,
        help="Enumeration (e.g., '6' or '3,4')",
    )
    parser.add_argument(
        "--num_clues",
        type=int,
        default=3,
        help="Number of clues to generate",
    )

    args = parser.parse_args()

    model, tokenizer, device = load_model(args.base_model, args.lora_path, args.device)

    if args.answer and args.enumeration:
        clues = generate_clue(
            model, tokenizer, args.answer.upper(), args.enumeration, device,
            num_return_sequences=args.num_clues,
        )
        print(f"\nGenerated clues for '{args.answer.upper()}' ({args.enumeration}):")
        if isinstance(clues, list):
            for i, clue in enumerate(clues, 1):
                print(f"  {i}. {clue}")
        else:
            print(f"  {clues}")
    else:
        interactive_mode(model, tokenizer, device)
