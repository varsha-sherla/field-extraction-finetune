"""
Scores model predictions against ground truth. Pure Python, no ML libraries,
no GPU -- runs anywhere, including plain local VS Code with no accelerator.

This is deliberately split from generation: the notebook (which needs the
GPU) generates predictions and dumps them to jsonl; this script only reads
text files and computes metrics. That split means the one number that
actually matters for your CV -- "fine-tuned beat base by X points" -- can be
reproduced by anyone cloning your repo without needing a GPU themselves,
which is exactly what you want a reviewer/interviewer to be able to do.

Usage:
    python src/evaluate.py \
        --test data/test.jsonl \
        --predictions results/predictions_base.jsonl \
        --label "Base model (Phi-4-mini, zero-shot)"

    python src/evaluate.py \
        --test data/test.jsonl \
        --predictions results/predictions_finetuned.jsonl \
        --label "Fine-tuned (LoRA)" \
        --compare_to results/metrics_base.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import FIELDS  # noqa: E402


def extract_json(raw_output: str):
    """
    Models -- especially the un-fine-tuned base model -- often wrap JSON in
    markdown fences, add a preamble ("Sure, here's the JSON:"), or trail off
    with extra commentary. Pull out the first {...} block and try to parse
    it rather than requiring an exact clean response.
    Returns a dict, or None if nothing parseable was found.
    """
    if raw_output is None:
        return None
    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize(value) -> str:
    """Loose normalization so trivial formatting differences don't count as
    wrong: case, surrounding whitespace, and 'null'/None both mean missing."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in ("null", "none", "n/a", ""):
        return ""
    return re.sub(r"\s+", " ", s)


def score(test_examples, predictions):
    """
    test_examples: list of {"input_text":..., "fields": {...}}
    predictions:   list of {"raw_output": "..."} in the same order
    """
    assert len(test_examples) == len(predictions), (
        f"Mismatched lengths: {len(test_examples)} test examples vs "
        f"{len(predictions)} predictions -- did you run inference on the "
        f"full test.jsonl, in order, with nothing skipped?"
    )

    field_correct = {f: 0 for f in FIELDS}
    n = len(test_examples)
    parse_failures = 0
    exact_match_records = 0

    per_example = []
    for ex, pred in zip(test_examples, predictions):
        gt = ex["fields"]
        parsed = extract_json(pred.get("raw_output", ""))
        if parsed is None:
            parse_failures += 1
            per_example.append({"parsed": False, "field_hits": {f: False for f in FIELDS}})
            continue

        field_hits = {}
        for f in FIELDS:
            hit = normalize(parsed.get(f)) == normalize(gt.get(f))
            field_hits[f] = hit
            if hit:
                field_correct[f] += 1
        if all(field_hits.values()):
            exact_match_records += 1
        per_example.append({"parsed": True, "field_hits": field_hits})

    return {
        "n_examples": n,
        "parse_failure_rate": parse_failures / n,
        "field_accuracy": {f: field_correct[f] / n for f in FIELDS},
        "record_exact_match_rate": exact_match_records / n,
        "per_example": per_example,
    }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_report(label, metrics):
    print(f"\n=== {label} ===")
    print(f"  Examples scored:        {metrics['n_examples']}")
    print(f"  JSON parse failure rate: {metrics['parse_failure_rate']:.1%}")
    print("  Field accuracy:")
    for f, acc in metrics["field_accuracy"].items():
        print(f"    {f:10s}: {acc:.1%}")
    print(f"  Full-record exact match: {metrics['record_exact_match_rate']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--save_json", default=None,
                     help="path to save this run's metrics as JSON, e.g. results/metrics_finetuned.json")
    ap.add_argument("--compare_to", default=None,
                     help="path to a previously saved metrics JSON (e.g. the base model's) to print a delta against")
    args = ap.parse_args()

    test_examples = load_jsonl(args.test)
    predictions = load_jsonl(args.predictions)
    metrics = score(test_examples, predictions)
    print_report(args.label, metrics)

    if args.compare_to:
        with open(args.compare_to) as f:
            other = json.load(f)
        print(f"\n  --- Delta vs {args.compare_to} ---")
        for f in FIELDS:
            d = metrics["field_accuracy"][f] - other["field_accuracy"][f]
            print(f"    {f:10s}: {d:+.1%}")
        d_exact = metrics["record_exact_match_rate"] - other["record_exact_match_rate"]
        print(f"    {'exact match':10s}: {d_exact:+.1%}")

    if args.save_json:
        to_save = {k: v for k, v in metrics.items() if k != "per_example"}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nSaved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
