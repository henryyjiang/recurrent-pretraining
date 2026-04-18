"""
Build a 30,000-sample dataset drawn from Huginn's actual pretraining sources.

Sources are sampled proportionally to their training weights (high-weight sources
like Wikipedia and Cosmopedia get more samples). Uses streaming to avoid
downloading full datasets.

Output: data/pretrain_samples.jsonl  (one {"text": ...} per line)

Usage:
    python scripts/build_pretrain_sample_dataset.py [--output data/pretrain_samples.jsonl] [--total 30000] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SOURCES = [
    # (hf_id, config, split, text_field, weight, label)
    # wikimedia/wikipedia replaces euirim/goodwiki (script-based, broken)
    ("wikimedia/wikipedia",           "20231101.en",   "train", "text", 4.0, "wikipedia"),
    ("HuggingFaceTB/smollm-corpus",   "cosmopedia-v2", "train", "text", 2.0, "cosmopedia"),
    # RedPajama-Data-1T is script-based and no longer loadable; weight redistributed below
    ("HuggingFaceTB/smollm-corpus",   "fineweb-edu-dedup", "train", "text", 1.0, "fineweb_edu"),
]


def stream_samples(hf_id, config, split, text_field, n, seed):
    """Stream n samples from a dataset without downloading the full thing."""
    ds = load_dataset(hf_id, config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    collected = []
    for row in ds:
        text = row.get(text_field, "")
        if text and len(text.strip()) > 100:
            collected.append(text.strip())
        if len(collected) >= n:
            break
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pretrain_samples.jsonl")
    parser.add_argument("--total", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_weight = sum(w for *_, w, _ in SOURCES)
    allocations = {label: max(1, round(args.total * w / total_weight))
                   for *_, w, label in SOURCES}

    # Adjust last bucket so total == args.total exactly
    labels = list(allocations.keys())
    diff = args.total - sum(allocations.values())
    allocations[labels[-1]] += diff

    print(f"Sample allocations: {allocations}")

    all_samples = []
    for hf_id, config, split, text_field, weight, label in SOURCES:
        n = allocations[label]
        print(f"  Streaming {n:,} samples from {label} ({hf_id})...")
        try:
            texts = stream_samples(hf_id, config, split, text_field, n, args.seed)
            for t in texts:
                all_samples.append({"text": t, "source": label})
            print(f"    Got {len(texts):,} samples")
        except Exception as e:
            print(f"    WARNING: failed to load {label}: {e}")

    random.shuffle(all_samples)
    with open(out_path, "w") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_samples):,} samples to {out_path}")
    source_counts = {}
    for item in all_samples:
        source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt:,}")


if __name__ == "__main__":
    main()
