"""
Quick CLI to try the fine-tuned model on one new piece of text -- useful
for a live demo in an interview, or just sanity-checking the adapter after
training. Needs the same GPU-side dependencies as generate_predictions.py.

Usage:
    python src/inference.py \
        --model unsloth/Phi-4-mini-instruct-bnb-4bit \
        --adapter outputs/lora_adapter \
        --text "GREENLEAF GROCERY GMBH ... TOTAL EUR 42.10 ..."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import build_prompt  # noqa: E402
from generate_predictions import load_model, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--text", required=True, help="raw messy receipt/invoice text")
    args = ap.parse_args()

    model, tokenizer, backend = load_model(args.model, args.adapter)
    prompt = build_prompt(args.text)
    output = generate(model, tokenizer, backend, prompt)
    print(output)


if __name__ == "__main__":
    main()
