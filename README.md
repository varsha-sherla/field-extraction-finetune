# Structured Field Extraction — LoRA Fine-Tuned Small Language Model

Fine-tunes a small open-source language model (Phi-4-mini, 3.8B) to extract
structured fields — `company`, `date`, `address`, `total` — as JSON from
messy, unstructured receipt/invoice text, and benchmarks the fine-tuned
model against the untouched base model on a held-out test set.

**Cost: $0.** No paid API calls anywhere in this pipeline. Training runs on
a free Colab/Kaggle T4 GPU via [Unsloth](https://github.com/unslothai/unsloth);
everything else (data prep, evaluation, scoring) runs on a plain CPU,
including a laptop with no GPU at all.

## Why this project

Structured data extraction from messy documents — invoices, receipts,
contracts, intake forms — is one of the most common real automation
problems companies actually pay to solve, and it's a task where a small
fine-tuned model's improvement over the untouched base model is easy to
measure honestly with real numbers (field-level accuracy), not a vibes-based
"looks better" judgment call.

## Architecture

```
messy input text  ──►  prompt template  ──►  LLM  ──►  JSON output
                                                          │
                        held-out test set  ──►  evaluate.py  ──►  field-level
                                                                   accuracy report
```

Two models are compared on the exact same test set and prompt:
1. **Base model, zero-shot** — Phi-4-mini with no fine-tuning, just asked
   nicely (via the prompt) to extract the fields.
2. **Fine-tuned model** — the same base weights + a small LoRA adapter
   trained on labeled examples of this exact task.

## Project layout

```
data/
  generate_synthetic_data.py   synthetic messy-receipt generator (no deps, no network — for pipeline sanity checks)
  prepare_sroie.py             pulls the real ICDAR2019-SROIE dataset (987 real receipts, CC-BY-4.0) from Hugging Face
  train.jsonl / test.jsonl     generated data lands here
notebooks/
  finetune_unsloth.ipynb       the one notebook that needs a GPU — training + generating predictions
src/
  prompt_template.py           the exact prompt format, shared by training, eval, and inference
  evaluate.py                  scores predictions vs ground truth — pure Python, no GPU needed
  generate_predictions.py      CLI version of prediction generation, for GPU machines outside Colab
  inference.py                 try the fine-tuned model on one new example
results/                       predictions and metrics land here after you run the notebook
```

## Setup — running this from VS Code with a free Colab GPU

1. Push this folder to a GitHub repo (`git init && git add . && git commit -m "init" && git push`).
2. In VS Code, install the **"Google Colab"** extension from the marketplace.
3. Open `notebooks/finetune_unsloth.ipynb` in VS Code.
4. Use VS Code's kernel picker (top right of the notebook) to select **Colab**, sign in with a Google account, and choose a **GPU (T4)** runtime.
5. In the notebook's first code cell, replace `REPO_URL` with your repo's clone URL.
6. Run the cells top to bottom. Everything — editing, running, debugging — happens inside VS Code; the compute runs on Colab's free GPU.

If the Colab extension gives you trouble, the fallback is opening the same
notebook directly at [colab.research.google.com](https://colab.research.google.com)
(File → Upload notebook) — the notebook itself doesn't change, just where
you click "Run."

## Running it step by step

**1. Get data.** Two options, both handled by cells inside the notebook:
- Real data (use this for your actual CV numbers): `python data/prepare_sroie.py` — needs `pip install datasets`, needs normal internet access (this step won't work on a heavily locked-down network, but Colab's own network is fine).
- Synthetic data (quick pipeline check only, not for your final numbers): `python data/generate_synthetic_data.py --n 150 --seed 42` — a small sample is already committed in this repo so `evaluate.py` has something to run against immediately.

**2. Train.** Run the notebook. It loads Phi-4-mini in 4-bit, generates baseline predictions with the untouched model, attaches a LoRA adapter, fine-tunes on `train.jsonl`, then generates predictions again with the fine-tuned model — all on the free GPU.

**3. Score.** The notebook's last cells call `src/evaluate.py` on both prediction files and print a side-by-side comparison. You can also re-run this exact scoring step later on your own laptop with no GPU at all, since it only reads the saved `results/*.jsonl` files:

```bash
python src/evaluate.py --test data/test.jsonl \
    --predictions results/predictions_base.jsonl --label "Base model"

python src/evaluate.py --test data/test.jsonl \
    --predictions results/predictions_finetuned.jsonl --label "Fine-tuned" \
    --compare_to results/metrics_base.json
```

## Filling in your real numbers

Once you've run the notebook, replace this section with your actual output
from `evaluate.py` — this is the part of the README an interviewer will
actually read closely, so use the real printed numbers, not placeholders:

| Metric | Base model | Fine-tuned | Δ |
|---|---|---|---|
| Company accuracy | TBD | TBD | TBD |
| Date accuracy | TBD | TBD | TBD |
| Address accuracy | TBD | TBD | TBD |
| Total accuracy | TBD | TBD | TBD |
| Full-record exact match | TBD | TBD | TBD |
| JSON parse failure rate | TBD | TBD | TBD |

## Limitations (worth stating honestly, not hiding)

- A quantized 3.8B model run on a free T4 GPU will not match a frontier
  model's raw accuracy — the point of this project is the *before/after
  delta at zero marginal cost*, not beating GPT-4.
- The SROIE dataset is receipts specifically; generalizing to other
  document types (contracts, forms) would need additional labeled data.
- Evaluation uses exact-match string comparison after light normalization
  (case, whitespace) — it will undercount genuinely correct answers that
  are formatted differently (e.g. "12/01/2025" vs "Jan 12, 2025"). Worth
  mentioning if asked, not worth over-engineering for a portfolio project.

## Suggested CV bullet (fill in the real numbers first)

> Designed, fine-tuned, and evaluated a local open-source LLM (LoRA/QLoRA,
> Phi-4-mini) for structured field extraction from unstructured receipt
> text, improving full-record exact-match accuracy from TBD% to TBD% over
> the zero-shot base model — zero API cost, trained on a free GPU,
> reproducible evaluation pipeline requiring no GPU to re-run.
