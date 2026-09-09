"""
Generates a synthetic messy-receipt dataset for the field-extraction project.

Why synthetic data exists at all, given the README points you at the real
SROIE dataset for the actual CV-worthy training run: this generator has no
dependencies beyond the standard library, runs instantly on any machine
(no network, no GPU, no Hugging Face account), and is deterministic via a
fixed seed. Use it to:
  - sanity-check the whole pipeline (prompt format, train/test split,
    evaluate.py's scoring logic) before you ever touch a GPU
  - as a fallback if HF access is flaky when you're setting this up

For the version you actually put on your CV, use --source sroie in
prepare_dataset.py (see the README) so the numbers you quote come from a
real, citable public dataset rather than synthetic data.

Usage:
    python data/generate_synthetic_data.py --n 150 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

COMPANIES = [
    "GREENLEAF GROCERY GMBH", "Bahnhof Cafe & Bakery", "TechMart Electronics",
    "Alpenblick Hotel Restaurant", "Sunrise Pharmacy", "CityBike Rentals",
    "Nordsee Fish House", "Blaue Blume Florist", "Kaufhaus Weber",
    "Cafe Zeit", "Berlin Bike Shop", "Grüner Punkt Recycling Center",
]
STREETS = [
    "Hauptstrasse 12", "Berliner Allee 45", "Karl-Marx-Strasse 7",
    "Bahnhofstrasse 3", "Lindenweg 88", "Goethestrasse 21", "Marktplatz 5",
]
CITIES = ["10115 Berlin", "01067 Dresden", "03046 Cottbus", "80331 München", "50667 Köln"]

# Small vocabulary of OCR-style noise: characters that look-alike scanners
# commonly confuse, used to corrupt a few characters per line.
CONFUSABLES = {"O": "0", "I": "1", "S": "5", "B": "8", "l": "1", "e": "3"}


def _noisy(s: str, rate: float, rng: random.Random) -> str:
    out = []
    for ch in s:
        if ch in CONFUSABLES and rng.random() < rate:
            out.append(CONFUSABLES[ch])
        else:
            out.append(ch)
    return "".join(out)


def make_example(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES)
    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.choice([2024, 2025, 2026])
    date_str = f"{day:02d}/{month:02d}/{year}"
    address = f"{rng.choice(STREETS)}, {rng.choice(CITIES)}"
    total = round(rng.uniform(3.5, 189.99), 2)
    total_str = f"{total:.2f}"

    fields = {
        "company": company,
        "date": date_str,
        "address": address,
        "total": total_str,
    }

    # Build a "messy OCR dump" the way real scanned-receipt text tends to
    # look: lines out of logical order, stray items, inconsistent casing,
    # a couple of confusable-character typos, no clean labels on everything.
    lines = [
        _noisy(company, 0.15, rng),
        "THANK YOU FOR SHOPPING",
        f"Item {rng.randint(1,9)}x  Misc Goods   {rng.uniform(1,20):.2f}",
        _noisy(address, 0.1, rng),
        f"Date: {date_str}" if rng.random() < 0.5 else date_str,
        f"VAT ID DE{rng.randint(100000000, 999999999)}",
        "-------------------------",
        f"TOTAL{'  ' if rng.random()<0.5 else ' EUR '}{total_str}",
        "CARD PAYMENT" if rng.random() < 0.5 else "CASH",
    ]
    rng.shuffle(lines[:5])  # shuffle only the "body" lines, keep total/footer near the end
    text = "\n".join(lines)
    return {"input_text": text, "fields": fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = [make_example(rng) for _ in range(args.n)]
    rng.shuffle(examples)
    n_test = max(1, int(len(examples) * args.test_frac))
    test, train = examples[:n_test], examples[n_test:]

    out_dir = Path(args.out_dir)
    for name, split in [("train.jsonl", train), ("test.jsonl", test)]:
        with open(out_dir / name, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split)} examples to {out_dir / name}")


if __name__ == "__main__":
    main()
