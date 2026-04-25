"""
CCoT Gaussian-noise ablation.

Identical setup to the original full-rank frozen CCoT run on winogrande_hellaswag,
except the latent state passed between CCoT passes is replaced with matched Gaussian
noise.  This tests whether the learned *content* of the latent chain matters, or
whether any injection (including noise) produces similar performance.

Noise is scaled to match the RMS norm of the actual latent state computed per-batch,
so the injected signal has the same energy as the real one.

Usage:
    python train_noise_ablation.py \\
        [--output_dir ./runs/noise_ablation] \\
        [--model_name tomg-group-umd/huginn-0125] \\
        [--hf_cache /scratch/$USER/hf_cache] \\
        [--noise_scale 1.0]          # multiplier on matched-norm noise
        [--fixed_noise_std 0.0]      # if >0, bypass norm-matching and use fixed std

Output layout:
    <output_dir>/
        ccot_noise/     ← checkpoint
        run_config.json
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str = ""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from recpre.raven_config_minimal import RavenConfig


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",        default="./runs/noise_ablation")
    p.add_argument("--model_name",        default="tomg-group-umd/huginn-0125")
    p.add_argument("--hf_cache",          default=None)
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--batch_size",        type=int,   default=1)
    p.add_argument("--grad_accum",        type=int,   default=32)
    p.add_argument("--num_steps",         type=int,   default=32)
    p.add_argument("--num_passes",        type=int,   default=4)
    p.add_argument("--backprop_depth",    type=int,   default=4)
    p.add_argument("--max_seq_len",       type=int,   default=256)
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--weight_decay",      type=float, default=0.1)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--log_interval",      type=int,   default=20)
    p.add_argument("--max_train_samples", type=int,   default=None)
    p.add_argument("--grad_checkpoint",   action="store_true")
    # Noise parameters
    p.add_argument("--noise_scale",   type=float, default=1.0,
                   help="Multiplier applied to the norm-matched Gaussian noise. "
                        "1.0 = same energy as actual latent.")
    p.add_argument("--fixed_noise_std", type=float, default=0.0,
                   help="If >0, skip norm-matching and inject N(0, fixed_noise_std) noise.")
    return p.parse_args()


ARGS       = None
DEVICE     = "cuda"
DTYPE      = torch.bfloat16
tokenizer  = None
OUTPUT_DIR = None


# ---------------------------------------------------------------------------
# Dataset  (winogrande + hellaswag, same as original best run)
# ---------------------------------------------------------------------------
def tokenize_text(text: str):
    ids     = tokenizer(text, add_special_tokens=True).input_ids[:ARGS.max_seq_len]
    pad_id  = tokenizer.pad_token_id or 0
    pad_len = ARGS.max_seq_len - len(ids)
    return (
        torch.tensor(ids + [pad_id] * pad_len),
        torch.tensor(ids + [-100]   * pad_len),
    )


class HellaSwagDataset(Dataset):
    def __init__(self):
        self.data = load_dataset("Rowan/hellaswag", split="train")

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        ex   = self.data[idx]
        text = f"{ex['ctx']} {ex['endings'][int(ex['label'])]}"
        iids, labs = tokenize_text(text)
        return {"input_ids": iids, "labels": labs}


class WinoGrandeDataset(Dataset):
    def __init__(self):
        self.data = load_dataset("winogrande", "winogrande_xl", split="train")

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        ex      = self.data[idx]
        correct = ex["option1"] if ex["answer"] == "1" else ex["option2"]
        text    = ex["sentence"].replace("_", correct)
        iids, labs = tokenize_text(text)
        return {"input_ids": iids, "labels": labs}


def build_dataset():
    per = ARGS.max_train_samples // 2 if ARGS.max_train_samples else None

    log("Loading HellaSwag train...")
    hs = HellaSwagDataset()
    if per:
        hs = Subset(hs, range(min(per, len(hs))))
    log(f"  {len(hs):,} examples")

    log("Loading WinoGrande train (xl)...")
    wg = WinoGrandeDataset()
    if per:
        wg = Subset(wg, range(min(per, len(wg))))
    log(f"  {len(wg):,} examples")

    combined = ConcatDataset([hs, wg])
    log(f"Combined: {len(combined):,} examples")
    return combined


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model():
    cfg = RavenConfig.from_pretrained(ARGS.model_name)
    cfg.iter_injection      = "none"
    cfg.ccot_injection      = "add"
    cfg.proj_bottleneck_dim = 0      # full-rank ccot_proj (same as original best run)

    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name,
        config=cfg,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        ignore_mismatched_sizes=True,
    )

    if ARGS.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Freeze transformer; only ccot_proj is trainable (mirrors original best run).
    for name, param in model.named_parameters():
        param.requires_grad = "ccot_proj" in name

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    return model


def save_checkpoint(model):
    path = OUTPUT_DIR / "ccot_noise"
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items()
             if any(p.requires_grad for n, p in model.named_parameters() if n == k)}
    torch.save(state, path / "trainable_weights.pt")
    model.config.save_pretrained(str(path))
    size_mb = sum(t.numel() * t.element_size() for t in state.values()) / 1e6
    log(f"Saved → {path}  ({sum(t.numel() for t in state.values()):,} params, {size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Noise helper
# ---------------------------------------------------------------------------
def make_noise(latent: torch.Tensor) -> torch.Tensor:
    """Return Gaussian noise of the same shape as `latent`.

    By default, noise is scaled to match the per-sample RMS norm of the actual
    latent so the injected energy equals the real latent energy (norm-matched).
    Set --fixed_noise_std > 0 to use a fixed standard deviation instead.
    """
    if ARGS.fixed_noise_std > 0:
        return torch.randn_like(latent) * ARGS.fixed_noise_std

    # Norm-matched: scale each sample so ||noise||_F == ||latent||_F
    noise      = torch.randn_like(latent)
    # Per-sample Frobenius norms over (S, E) dims
    lat_norm   = latent.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)   # [B,1,1]
    noise_norm = noise.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)    # [B,1,1]
    return noise * (lat_norm / noise_norm) * ARGS.noise_scale


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model):
    model.train()

    ds     = build_dataset()
    loader = DataLoader(ds, batch_size=ARGS.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)

    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(trainable, lr=ARGS.lr,
                                     weight_decay=ARGS.weight_decay)
    total_steps  = len(loader) * ARGS.epochs // ARGS.grad_accum
    warmup_steps = max(1, int(total_steps * ARGS.warmup_ratio))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    steps_per_pass  = max(1, ARGS.num_steps // ARGS.num_passes)
    no_grad_p       = max(0, steps_per_pass - ARGS.backprop_depth)
    with_grad_p     = steps_per_pass - no_grad_p
    num_steps_pass  = torch.tensor([no_grad_p, with_grad_p])

    output_details = {
        "return_logits": False, "return_latents": True,
        "return_head":   False, "return_stats":   False,
    }

    log(f"CCoT noise ablation: {ARGS.num_passes} passes × {steps_per_pass} steps/pass")
    log(f"Norm-matched noise scale: {ARGS.noise_scale} | fixed std: {ARGS.fixed_noise_std}")
    log(f"Epochs={ARGS.epochs}  steps/epoch={len(loader)}  "
        f"eff_batch={ARGS.batch_size * ARGS.grad_accum}  LR={ARGS.lr}")

    global_step = 0
    accum_loss  = 0.0
    accum_count = 0

    for epoch in range(ARGS.epochs):
        for batch_idx, batch in enumerate(loader):
            iids   = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

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
                # Replace the actual latent with matched noise; the ccot_proj
                # therefore never sees real learned representations.
                latent      = out.latent_states.detach()
                ccot_memory = make_noise(latent)

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
                    log(f"epoch={epoch+1}  step={global_step}  "
                        f"loss={avg:.4f}  lr={lr:.2e}")
                    accum_loss  = 0.0
                    accum_count = 0

        log(f"── Epoch {epoch+1} complete ──")

    model.eval()
    save_checkpoint(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global ARGS, tokenizer, OUTPUT_DIR

    ARGS = parse_args()

    if ARGS.hf_cache:
        os.environ["HF_HOME"]            = ARGS.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = ARGS.hf_cache
        os.environ["HF_DATASETS_CACHE"]  = ARGS.hf_cache

    OUTPUT_DIR = Path(ARGS.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"torch {torch.__version__} | device: {DEVICE} | dtype: {DTYPE}")
    log(f"Output dir: {OUTPUT_DIR}")

    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(vars(ARGS), indent=2, default=str)
    )

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name)

    log("Loading model...")
    model = load_model()

    train(model)

    log("Done.")


if __name__ == "__main__":
    main()
