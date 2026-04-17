"""
Huginn CCoT finetuning — HPC script version.

Trains up to three variants of Huginn on GSM8K, evaluates all against the frozen
baseline, and writes results to an output directory.

CCoT design (ccot_only / both):
  Each training example is processed in NUM_PASSES sequential full-transformer
  passes, each using ARGS.num_steps // ARGS.num_passes recurrent steps.  The
  final loop-block latent state from pass k is detached and injected (via
  ccot_proj) at the start of every recurrent step of pass k+1.  Loss is only
  computed on the last pass.  This mirrors latent chain-of-thought: the model
  re-reads the same question multiple times, refining its internal state before
  committing to an answer — analogous to CoT token generation but without
  materialising any tokens.

Usage:
    python train_ccot.py [--experiments iter_only ccot_only both] \\
                         [--output_dir ./runs/ccot_001] \\
                         [--model_name tomg-group-umd/huginn-0125] \\
                         [--hf_cache /scratch/$USER/hf_cache] \\
                         [--epochs 5] [--lr 3e-4] [--num_passes 4] \\
                         [--no_train_loop] [--skip_eval] [--skip_qualitative]

Output layout:
    <output_dir>/
        iter_only/          ← saved checkpoint (trainable_weights.pt + config)
        ccot_only/
        both/
        results.csv
        results.txt
        accuracy_vs_steps.png
        qual_outputs.txt
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str = ""):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Bootstrap: load the two minimal modeling files WITHOUT triggering
# recpre/__init__.py (which imports a torch internal symbol removed in 2.11).
# ---------------------------------------------------------------------------
def _bootstrap_recpre(root: Path):
    for _name, _rel in [
        ("recpre.raven_config_minimal",   "recpre/raven_config_minimal.py"),
        ("recpre.raven_modeling_minimal", "recpre/raven_modeling_minimal.py"),
    ]:
        _path = root / _rel
        if not _path.exists():
            raise FileNotFoundError(f"Cannot find {_path}. Run from repo root.")
        _spec = importlib.util.spec_from_file_location(_name, str(_path))
        _mod  = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
_bootstrap_recpre(REPO_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from recpre.raven_config_minimal import RavenConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Huginn CCoT finetuning")
    p.add_argument(
        "--experiments", nargs="+",
        choices=["iter_only", "ccot_only", "both"],
        default=["iter_only", "ccot_only", "both"],
        help="Which experiments to run (default: all three)",
    )
    p.add_argument("--output_dir",  default="./runs/ccot",
                   help="Root directory for checkpoints and results")
    p.add_argument("--model_name",  default="tomg-group-umd/huginn-0125")
    p.add_argument("--hf_cache",    default=None,
                   help="HuggingFace cache dir (set to a scratch path on HPC)")
    # Training hyperparams
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--grad_accum",      type=int,   default=16)
    p.add_argument("--num_steps",       type=int,   default=32,
                   help="Total recurrent steps per forward pass (or total across all passes for CCoT)")
    p.add_argument("--num_passes",      type=int,   default=4,
                   help="Number of full-transformer passes per example for CCoT experiments. "
                        "steps_per_pass = num_steps // num_passes")
    p.add_argument("--backprop_depth",  type=int,   default=8,
                   help="Recurrent steps with gradient per pass (TBPTT depth)")
    p.add_argument("--max_seq_len",     type=int,   default=256)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--log_interval",    type=int,   default=20)
    # Flags
    p.add_argument("--no_train_loop",   action="store_true",
                   help="Freeze core_block (train proj weights only)")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Enable gradient checkpointing (saves memory, slower)")
    # Eval
    p.add_argument("--dataset",         default="winogrande_hellaswag",
                   choices=["gsm8k", "winogrande_hellaswag"],
                   help="Training dataset. winogrande_hellaswag mixes both train splits "
                        "for broader coverage and reduced task-specific overfitting.")
    p.add_argument("--skip_eval",       action="store_true")
    p.add_argument("--skip_qualitative",action="store_true")
    p.add_argument("--n_eval",          type=int,   default=200)
    p.add_argument("--eval_steps",      nargs="+",  type=int,
                   default=[4, 8, 16, 32])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Global state (set in main after args are parsed)
# ---------------------------------------------------------------------------
ARGS        = None
DEVICE      = "cuda"
DTYPE       = torch.bfloat16
tokenizer   = None
raw         = None   # HuggingFace dataset dict (GSM8K only, for eval)
OUTPUT_DIR  = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def tokenize_text(text: str):
    """Causal LM tokenization: loss on all non-padding tokens."""
    ids     = tokenizer(text, add_special_tokens=True).input_ids[:ARGS.max_seq_len]
    pad_id  = tokenizer.pad_token_id or 0
    pad_len = ARGS.max_seq_len - len(ids)
    input_ids = ids + [pad_id] * pad_len
    labels    = ids + [-100]   * pad_len
    return torch.tensor(input_ids), torch.tensor(labels)


def tokenize_example(q: str, a: str):
    """GSM8K tokenization: loss on answer tokens only."""
    text   = f"Question: {q.strip()}\nAnswer: {a.split('####')[-1].strip()}"
    q_text = f"Question: {q.strip()}\nAnswer:"

    full_ids = tokenizer(text,   add_special_tokens=True).input_ids[:ARGS.max_seq_len]
    q_ids    = tokenizer(q_text, add_special_tokens=True).input_ids

    labels  = [-100] * len(q_ids) + full_ids[len(q_ids):]
    labels  = labels[:ARGS.max_seq_len]

    pad_id  = tokenizer.pad_token_id or 0
    pad_len = ARGS.max_seq_len - len(full_ids)
    input_ids = full_ids + [pad_id] * pad_len
    labels    = labels   + [-100]   * pad_len
    return torch.tensor(input_ids), torch.tensor(labels)


class GSM8KDataset(Dataset):
    def __init__(self, split="train"):
        self.data = raw[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        iids, labs = tokenize_example(row["question"], row["answer"])
        return {"input_ids": iids, "labels": labs}


class HellaSwagDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("Rowan/hellaswag", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex      = self.data[idx]
        correct = ex["endings"][int(ex["label"])]
        text    = f"{ex['ctx']} {correct}"
        iids, labs = tokenize_text(text)
        return {"input_ids": iids, "labels": labs}


class WinoGrandeDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("winogrande", "winogrande_xl", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex      = self.data[idx]
        correct = ex["option1"] if ex["answer"] == "1" else ex["option2"]
        text    = ex["sentence"].replace("_", correct)
        iids, labs = tokenize_text(text)
        return {"input_ids": iids, "labels": labs}


def build_train_dataset() -> Dataset:
    if ARGS.dataset == "gsm8k":
        return GSM8KDataset("train")
    # winogrande_hellaswag: ~80K diverse examples across two tasks
    from torch.utils.data import ConcatDataset
    log("  Loading HellaSwag train...")
    hs = HellaSwagDataset("train")
    log(f"    {len(hs):,} examples")
    log("  Loading WinoGrande train (xl)...")
    wg = WinoGrandeDataset("train")
    log(f"    {len(wg):,} examples")
    combined = ConcatDataset([hs, wg])
    log(f"  Combined: {len(combined):,} examples")
    return combined


def collate_single(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def load_model(iter_injection="none", ccot_injection="none", train_loop=True):
    cfg = RavenConfig.from_pretrained(ARGS.model_name)
    cfg.iter_injection = iter_injection
    cfg.ccot_injection = ccot_injection

    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name,
        config=cfg,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        ignore_mismatched_sizes=True,
    )

    if ARGS.grad_checkpoint:
        model.gradient_checkpointing_enable()

    trainable_keywords = []
    if iter_injection != "none":
        trainable_keywords.append("iter_proj")
    if ccot_injection != "none":
        trainable_keywords.append("ccot_proj")
    if train_loop:
        trainable_keywords.append("core_block")

    for name, param in model.named_parameters():
        param.requires_grad = any(kw in name for kw in trainable_keywords)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.2f}%)")
    log(f"  iter_injection={iter_injection!r}  ccot_injection={ccot_injection!r}  "
        f"train_loop={train_loop}")
    return model


def save_checkpoint(model, name: str):
    """Save only the trainable weights + config."""
    path = OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)

    trainable_state = {k: v for k, v in model.state_dict().items()
                       if any(p.requires_grad for n, p in model.named_parameters() if n == k)}
    torch.save(trainable_state, path / "trainable_weights.pt")
    model.config.save_pretrained(str(path))

    n_params = sum(t.numel() for t in trainable_state.values())
    size_mb  = sum(t.numel() * t.element_size() for t in trainable_state.values()) / 1e6
    log(f"  Saved → {path}  ({n_params:,} params, {size_mb:.0f} MB)")


def load_checkpoint(name: str, iter_injection="none", ccot_injection="none"):
    """Load base model from HF cache, then overlay the saved trainable weights."""
    path = OUTPUT_DIR / name
    cfg  = RavenConfig.from_pretrained(str(path))
    cfg.iter_injection = iter_injection
    cfg.ccot_injection = ccot_injection

    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name, config=cfg, torch_dtype=DTYPE,
        device_map=DEVICE, ignore_mismatched_sizes=True,
    )

    weights_path = path / "trainable_weights.pt"
    if weights_path.exists():
        delta = torch.load(str(weights_path), map_location=DEVICE)
        missing, unexpected = model.load_state_dict(delta, strict=False)
        if unexpected:
            log(f"  WARNING: unexpected keys in checkpoint: {unexpected}")
        log(f"  Loaded {len(delta)} weight tensors from {weights_path}")
    else:
        log(f"  WARNING: no trainable_weights.pt found at {path}, using base model")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, experiment_name: str, use_multipass: bool):
    """
    use_multipass=False  →  iter_only: single pass, num_steps total, no ccot.
    use_multipass=True   →  ccot_only / both: ARGS.num_passes sequential passes
                            over the same example, each with steps_per_pass
                            recurrent steps.  ccot_memory is chained between
                            passes (detached).  Loss on final pass only.
    """
    model.train()

    ds     = build_train_dataset()
    loader = DataLoader(ds, batch_size=ARGS.batch_size, shuffle=True,
                        collate_fn=collate_single, drop_last=True)

    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(trainable, lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    total_steps  = len(loader) * ARGS.epochs // ARGS.grad_accum
    warmup_steps = max(1, int(total_steps * ARGS.warmup_ratio))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # TBPTT split for a single pass (iter_only uses full num_steps)
    num_steps_single = torch.tensor(
        [ARGS.num_steps - ARGS.backprop_depth, ARGS.backprop_depth]
    )

    # Per-pass step counts for CCoT (steps_per_pass = num_steps // num_passes)
    if use_multipass:
        steps_per_pass = max(1, ARGS.num_steps // ARGS.num_passes)
        no_grad_pass   = max(0, steps_per_pass - ARGS.backprop_depth)
        with_grad_pass = steps_per_pass - no_grad_pass
        num_steps_pass = torch.tensor([no_grad_pass, with_grad_pass])
        log(f"  CCoT: {ARGS.num_passes} passes × {steps_per_pass} steps/pass "
            f"(no-grad={no_grad_pass}, grad={with_grad_pass} per pass)")
    else:
        log(f"  Single pass: {ARGS.num_steps} steps "
            f"(no-grad={ARGS.num_steps - ARGS.backprop_depth}, "
            f"grad={ARGS.backprop_depth})")

    # output_details: always return latents (needed for ccot chaining)
    output_details = {
        "return_logits": False, "return_latents": True,
        "return_head":   False, "return_stats":   False,
    }

    global_step  = 0
    accum_loss   = 0.0
    accum_count  = 0

    log(f"\n{'═'*60}")
    log(f"  Training: {experiment_name}")
    log(f"{'═'*60}")
    log(f"  Epochs={ARGS.epochs}  Steps/epoch={len(loader)}  "
        f"Effective batch={ARGS.batch_size * ARGS.grad_accum}  LR={ARGS.lr}")

    for epoch in range(ARGS.epochs):
        for batch_idx, batch in enumerate(loader):
            iids   = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if not use_multipass:
                # ── iter_only: single standard pass ──────────────────────────
                out  = model(
                    input_ids=iids, labels=labels,
                    num_steps=num_steps_single,
                    output_details=output_details,
                )
                loss = out.loss / ARGS.grad_accum

            else:
                # ── CCoT: num_passes sequential passes on the same example ──
                # Pass 0 .. num_passes-2: no labels, save latent → ccot_memory
                # Pass num_passes-1: labels provided, compute loss
                # Gradient is cut between passes (detach); each pass only
                # backprops through its own recurrent steps (TBPTT via
                # num_steps_pass).
                ccot_memory = None
                for p in range(ARGS.num_passes):
                    is_final = (p == ARGS.num_passes - 1)
                    out = model(
                        input_ids=iids,
                        labels=labels if is_final else None,
                        num_steps=num_steps_pass,
                        ccot_memory=ccot_memory,
                        output_details=output_details,
                    )
                    # Detach: gradients do not flow back through pass boundaries.
                    # out.latent_states is always populated (set before loss
                    # computation in the model's forward, independent of labels).
                    ccot_memory = out.latent_states.detach()
                loss = out.loss / ARGS.grad_accum

            loss.backward()
            accum_loss  += loss.item() * ARGS.grad_accum
            accum_count += 1

            if (batch_idx + 1) % ARGS.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, ARGS.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % ARGS.log_interval == 0:
                    avg = accum_loss / accum_count
                    lr  = scheduler.get_last_lr()[0]
                    log(f"  epoch={epoch+1}  step={global_step}  "
                        f"loss={avg:.4f}  lr={lr:.2e}")
                    accum_loss  = 0.0
                    accum_count = 0

        log(f"  ── Epoch {epoch+1} complete ──")

    model.eval()
    save_checkpoint(model, experiment_name)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def extract_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


@torch.no_grad()
def evaluate_model(model, use_multipass: bool = False):
    """
    use_multipass=True:  for each example, run num_passes-1 warmup passes
                         (chaining ccot_memory), then generate on the final
                         pass with the accumulated memory.  ccot_memory resets
                         between examples — there is no cross-question chaining.
    """
    test_data = raw["test"].select(range(ARGS.n_eval))
    results   = {s: {"correct": 0, "total": 0} for s in ARGS.eval_steps}

    latent_od = {"return_logits": False, "return_latents": True,
                 "return_head":   False, "return_stats":   False}

    for num_steps in ARGS.eval_steps:
        steps_per_pass = (max(1, num_steps // ARGS.num_passes)
                          if use_multipass else num_steps)
        log(f"    evaluating steps={num_steps} "
            f"({'×'.join([str(ARGS.num_passes), str(steps_per_pass)]) if use_multipass else 'single pass'}) ...")

        for row in test_data:
            gold   = extract_number(row["answer"].split("####")[-1])
            prompt = f"Question: {row['question'].strip()}\nAnswer:"

            enc   = tokenizer(prompt, return_tensors="pt")
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc["attention_mask"].to(DEVICE)

            if use_multipass:
                # Warmup passes (no generation, just build up ccot_memory)
                ccot_memory = None
                for _ in range(ARGS.num_passes - 1):
                    fwd = model(
                        input_ids=iids,
                        num_steps=steps_per_pass,
                        ccot_memory=ccot_memory,
                        output_details=latent_od,
                    )
                    ccot_memory = fwd.latent_states.detach()

                # Final pass: generate with accumulated ccot_memory
                out = model.generate(
                    input_ids=iids, attention_mask=amask,
                    max_new_tokens=32, num_steps=steps_per_pass,
                    ccot_memory=ccot_memory, do_sample=False,
                )
            else:
                out = model.generate(
                    input_ids=iids, attention_mask=amask,
                    max_new_tokens=32, num_steps=num_steps, do_sample=False,
                )

            generated = tokenizer.decode(out[0][iids.shape[1]:], skip_special_tokens=True)
            pred      = extract_number(generated)

            results[num_steps]["correct"] += int(pred == gold)
            results[num_steps]["total"]   += 1

    return {s: r["correct"] / r["total"] for s, r in results.items()}


# ---------------------------------------------------------------------------
# Qualitative
# ---------------------------------------------------------------------------
QUAL_PROMPTS = [
    "Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nA:",
    "Q: Sarah has 24 apples. She gives half to Tom, then buys 6 more. How many does she have?\nA:",
    "Q: If it takes 5 machines 5 minutes to make 5 widgets, how long does 1 machine take to make 1 widget?\nA:",
]


@torch.no_grad()
def quick_generate(model, prompt: str, num_steps: int = 16,
                   use_multipass: bool = False) -> str:
    enc   = tokenizer(prompt, return_tensors="pt")
    iids  = enc["input_ids"].to(DEVICE)
    amask = enc["attention_mask"].to(DEVICE)

    if use_multipass:
        steps_per_pass = max(1, num_steps // ARGS.num_passes)
        latent_od = {"return_logits": False, "return_latents": True,
                     "return_head":   False, "return_stats":   False}
        ccot_memory = None
        for _ in range(ARGS.num_passes - 1):
            fwd = model(input_ids=iids, num_steps=steps_per_pass,
                        ccot_memory=ccot_memory, output_details=latent_od)
            ccot_memory = fwd.latent_states.detach()
        out = model.generate(input_ids=iids, attention_mask=amask,
                             max_new_tokens=40, num_steps=steps_per_pass,
                             ccot_memory=ccot_memory, do_sample=False)
    else:
        out = model.generate(input_ids=iids, attention_mask=amask,
                             max_new_tokens=40, num_steps=num_steps,
                             do_sample=False)

    return tokenizer.decode(out[0][iids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------
def save_results(all_results: dict):
    steps = ARGS.eval_steps
    col_w = 10
    header = f"{'Model':<16}" + "".join(f"{'steps='+str(s):>{col_w}}" for s in steps)
    lines  = [header, "-" * len(header)]
    for label, res in all_results.items():
        row = f"{label:<16}" + "".join(f"{res[s]*100:>{col_w}.1f}%" for s in steps)
        lines.append(row)
    table = "\n".join(lines)

    log("\n" + table)

    txt_path = OUTPUT_DIR / "results.txt"
    txt_path.write_text(table + "\n")
    log(f"  Saved → {txt_path}")

    try:
        import csv
        csv_path = OUTPUT_DIR / "results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Model"] + [f"steps={s}" for s in steps])
            for label, res in all_results.items():
                w.writerow([label] + [f"{res[s]*100:.1f}" for s in steps])
        log(f"  Saved → {csv_path}")
    except Exception as e:
        log(f"  WARNING: CSV save failed: {e}")

    json_path = OUTPUT_DIR / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    log(f"  Saved → {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        styles = {
            "Baseline":  ("black",     "--"),
            "iter_only": ("royalblue", "-"),
            "ccot_only": ("tomato",    "-"),
            "both":      ("seagreen",  "-"),
        }
        for label, res in all_results.items():
            xs = sorted(res.keys())
            ys = [res[x] * 100 for x in xs]
            color, ls = styles.get(label, ("gray", "-"))
            ax.plot(xs, ys, marker="o", label=label, color=color, linestyle=ls)

        ax.set_xlabel("num_steps (total recurrent steps)")
        ax.set_ylabel("Accuracy (%) on GSM8K test")
        ax.set_title("Huginn CCoT — Accuracy vs Recurrence Depth")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = OUTPUT_DIR / "accuracy_vs_steps.png"
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
        log(f"  Saved → {png_path}")
    except Exception as e:
        log(f"  WARNING: Plot save failed: {e}")


def save_qualitative(qual_lines: list[str]):
    path = OUTPUT_DIR / "qual_outputs.txt"
    path.write_text("\n".join(qual_lines) + "\n")
    log(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global ARGS, tokenizer, raw, OUTPUT_DIR

    ARGS = parse_args()

    if ARGS.hf_cache:
        os.environ["HF_HOME"]            = ARGS.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = ARGS.hf_cache
        os.environ["HF_DATASETS_CACHE"]  = ARGS.hf_cache

    OUTPUT_DIR = Path(ARGS.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"torch {torch.__version__} | device: {DEVICE} | dtype: {DTYPE}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Experiments: {ARGS.experiments}")
    log(f"train_loop: {not ARGS.no_train_loop}")
    log(f"num_passes: {ARGS.num_passes}  steps_per_pass: {ARGS.num_steps // ARGS.num_passes}")

    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(vars(ARGS), indent=2, default=str)
    )

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name)

    log(f"Dataset: {ARGS.dataset}")
    if ARGS.dataset == "gsm8k" or not ARGS.skip_eval:
        log("Loading GSM8K (needed for eval)...")
        raw = load_dataset("openai/gsm8k", "main")
        log(f"  Train: {len(raw['train'])}  |  Test: {len(raw['test'])}")

    train_loop = not ARGS.no_train_loop

    # ── Experiment definitions ─────────────────────────────────────────────
    # (iter_injection, ccot_injection, use_multipass)
    #   use_multipass=True  → num_passes sequential passes over one example,
    #                         ccot_memory chained between passes (single-problem
    #                         latent CoT).
    #   use_multipass=False → standard single-pass (iter_only).
    experiment_configs = {
        "iter_only":  ("add",  "none", False),
        "ccot_only":  ("none", "add",  True),
        "both":       ("add",  "add",  True),
    }

    # ── Training ──────────────────────────────────────────────────────────
    for name in ARGS.experiments:
        iter_inj, ccot_inj, use_multipass = experiment_configs[name]
        log(f"\nLoading model for {name}...")
        model = load_model(iter_inj, ccot_inj, train_loop=train_loop)
        train(model, name, use_multipass)
        del model
        torch.cuda.empty_cache()
        log(f"{name} done.")

    # ── Evaluation ────────────────────────────────────────────────────────
    if not ARGS.skip_eval:
        log("\n" + "═" * 60)
        log("  EVALUATION")
        log("═" * 60)

        eval_experiments = [("Baseline", "none", "none", None, False)]
        for name in ARGS.experiments:
            iter_inj, ccot_inj, use_multipass = experiment_configs[name]
            eval_experiments.append((name, iter_inj, ccot_inj, name, use_multipass))

        all_results = {}
        for label, iter_inj, ccot_inj, ckpt_name, use_multipass in eval_experiments:
            log(f"\nEvaluating: {label}")

            if ckpt_name is None:
                cfg   = RavenConfig.from_pretrained(ARGS.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    ARGS.model_name, config=cfg, torch_dtype=DTYPE,
                    device_map=DEVICE, ignore_mismatched_sizes=True,
                )
                model.eval()
            else:
                model = load_checkpoint(ckpt_name, iter_inj, ccot_inj)

            res = evaluate_model(model, use_multipass=use_multipass)
            all_results[label] = res

            for steps, acc in res.items():
                log(f"  steps={steps:2d}  acc={acc*100:.1f}%")

            del model
            torch.cuda.empty_cache()

        log("\nSaving results...")
        save_results(all_results)

    # ── Qualitative ───────────────────────────────────────────────────────
    if not ARGS.skip_qualitative:
        log("\n" + "═" * 60)
        log("  QUALITATIVE COMPARISON (num_steps=16)")
        log("═" * 60)

        qual_experiments = [("Baseline", "none", "none", None, False)]
        for name in ARGS.experiments:
            iter_inj, ccot_inj, use_multipass = experiment_configs[name]
            qual_experiments.append((name, iter_inj, ccot_inj, name, use_multipass))

        qual_lines = []
        for prompt in QUAL_PROMPTS:
            header = f"PROMPT: {prompt.splitlines()[0]}"
            log(header)
            log("-" * 70)
            qual_lines.extend([header, "-" * 70])

            for label, iter_inj, ccot_inj, ckpt_name, use_multipass in qual_experiments:
                if ckpt_name is None:
                    cfg = RavenConfig.from_pretrained(ARGS.model_name)
                    m   = AutoModelForCausalLM.from_pretrained(
                        ARGS.model_name, config=cfg, torch_dtype=DTYPE,
                        device_map=DEVICE, ignore_mismatched_sizes=True,
                    )
                else:
                    m = load_checkpoint(ckpt_name, iter_inj, ccot_inj)
                m.eval()

                ans  = quick_generate(m, prompt, num_steps=16,
                                      use_multipass=use_multipass)
                line = f"  [{label:<12}] {ans[:120]}"
                log(line)
                qual_lines.append(line)

                del m
                torch.cuda.empty_cache()

            qual_lines.append("")
            log("")

        save_qualitative(qual_lines)

    log("\nAll done.")


if __name__ == "__main__":
    main()
