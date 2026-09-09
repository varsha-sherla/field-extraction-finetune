"""
Converts the real ICDAR2019-SROIE receipt dataset (Hugging Face:
jsdnrs/ICDAR2019-SROIE, CC-BY-4.0, 987 receipts) into this project's
train.jsonl / test.jsonl format.

This is the dataset to use for the version of the project that goes on your
CV and GitHub — it's real scanned-receipt OCR text with human-annotated
ground truth, not synthetic. Run this from the Colab notebook (or anywhere
with normal internet access) rather than a locked-down sandbox: it needs to
reach huggingface.co, which some restricted networks block.

Each SROIE example has:
  - "words": a list of OCR-extracted text tokens, in roughly the order the
    OCR engine read them off the receipt image (this is naturally "messy" —
    out of logical reading order, no punctuation cleanup)
  - "entities": a dict with the ground-truth fields we care about
    (company, date, address, total)

We join "words" into a single text blob to stand in for "raw messy OCR
output," and use "entities" directly as the target JSON — matching the
FIELDS defined in src/prompt_template.py.

Usage (needs `pip install datasets`):
    python data/prepare_sroie.py --test_frac 0.2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from prompt_template import FIELDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("jsdnrs/ICDAR2019-SROIE")
    out_dir = Path(args.out_dir)

    for hf_split, out_name in [("train", "train.jsonl"), ("test", "test.jsonl")]:
        if hf_split not in ds:
            print(f"Warning: split '{hf_split}' not found, skipping", file=sys.stderr)
            continue
        n_written = 0
        with open(out_dir / out_name, "w") as f:
            for row in ds[hf_split]:
                words = row.get("words") or []
                entities = row.get("entities") or {}
                input_text = " ".join(str(w) for w in words).strip()
                fields = {k: entities.get(k) for k in FIELDS}
                # Skip rows where every field is missing -- not useful signal.
                if not input_text or all(v is None for v in fields.values()):
                    continue
                f.write(json.dumps({"input_text": input_text, "fields": fields}) + "\n")
                n_written += 1
        print(f"Wrote {n_written} examples to {out_dir / out_name}")


if __name__ == "__main__":
    main()
