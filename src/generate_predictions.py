"""
Runs a model (base, or base + LoRA adapter) over test.jsonl and writes its
raw text output per example to a predictions jsonl that evaluate.py can
score. This is the one script in the repo that needs a GPU (or a lot of
patience on CPU) -- run it from the Colab notebook, or locally if you have
an NVIDIA GPU with enough VRAM.

Usage (base model, no adapter -- your "before" numbers):
    python src/generate_predictions.py \
        --model unsloth/Phi-4-mini-instruct-bnb-4bit \
        --test data/test.jsonl \
        --out results/predictions_base.jsonl

Usage (your fine-tuned adapter -- your "after" numbers):
    python src/generate_predictions.py \
        --model unsloth/Phi-4-mini-instruct-bnb-4bit \
        --adapter outputs/lora_adapter \
        --test data/test.jsonl \
        --out results/predictions_finetuned.jsonl

Swap --model for unsloth/Qwen2.5-3.5B-Instruct-bnb-4bit (or whichever exact
tag is current on the Unsloth Hugging Face org page at the time you run
this) if you went with Qwen instead of Phi-4-mini.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import build_prompt  # noqa: E402


def load_model(model_name: str, adapter_path: str | None, max_seq_length: int = 2048):
    """
    Tries Unsloth first (fast, matches the training notebook exactly).
    Falls back to plain transformers+peft if Unsloth isn't installed --
    useful if you're running this step somewhere other than the Colab
    notebook (e.g. a machine with its own GPU) and don't want the extra
    dependency.
    """
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return model, tokenizer, "transformers"


def generate(model, tokenizer, backend: str, prompt: str, max_new_tokens: int = 200) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, "device") else "cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministic -- we want a repeatable eval, not creative variety
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # The model's output includes our prompt; keep only what it generated after it.
    return full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model name/path, e.g. unsloth/Phi-4-mini-instruct-bnb-4bit")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter (omit for base-model-only run)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    model, tokenizer, backend = load_model(args.model, args.adapter)
    print(f"Loaded {args.model} via {backend} backend"
          + (f" + adapter {args.adapter}" if args.adapter else " (no adapter -- base model)"))

    test_examples = [json.loads(l) for l in open(args.test) if l.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        for i, ex in enumerate(test_examples):
            prompt = build_prompt(ex["input_text"])
            raw_output = generate(model, tokenizer, backend, prompt, args.max_new_tokens)
            f.write(json.dumps({"raw_output": raw_output}) + "\n")
            if (i + 1) % 10 == 0 or (i + 1) == len(test_examples):
                print(f"  {i + 1}/{len(test_examples)}")

    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
