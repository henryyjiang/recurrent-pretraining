"""
LoRA fine-tuning of Huginn core_block on winogrande_hellaswag.

Applies low-rank adapters (LoRA) directly to the recurrent core_block's
attention (Wqkv, proj) and MLP (fc, proj) layers, leaving all base weights
frozen.  This is a parameter-efficient alternative to the CCoT bottleneck
projection approach: instead of a separate latent-chaining module, we adapt
the transformer weights themselves.

No CCoT multipass — standard single-pass causal LM training.

Usage:
    python train_lora.py \\
        [--output_dir ./runs/lora] \\
        [--lora_rank 32] \\
        [--lora_alpha 32] \\
        [--lora_targets Wqkv,proj,fc] \\
        [--model_name tomg-group-umd/huginn-0125] \\
        [--hf_cache /scratch/$USER/hf_cache]

Output layout:
    <output_dir>/
        lora/   ← checkpoint (lora_weights.pt + config)
        run_config.json
"""

import argparse
import importlib.util
import json
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
import torch.nn as nn
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
    p.add_argument("--output_dir",        default="./runs/lora")
    p.add_argument("--model_name",        default="tomg-group-umd/huginn-0125")
    p.add_argument("--hf_cache",          default=None)
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--batch_size",        type=int,   default=1)
    p.add_argument("--grad_accum",        type=int,   default=32)
    p.add_argument("--num_steps",         type=int,   default=32,
                   help="Recurrent steps per forward pass")
    p.add_argument("--backprop_depth",    type=int,   default=4)
    p.add_argument("--max_seq_len",       type=int,   default=256)
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--weight_decay",      type=float, default=0.1)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--log_interval",      type=int,   default=20)
    p.add_argument("--max_train_samples", type=int,   default=None)
    p.add_argument("--grad_checkpoint",   action="store_true")
    # LoRA
    p.add_argument("--lora_rank",    type=int,   default=32,
                   help="LoRA rank r. Parameter count ∝ 2*r per target layer.")
    p.add_argument("--lora_alpha",   type=float, default=None,
                   help="LoRA scaling alpha (defaults to lora_rank, giving scale=1).")
    p.add_argument("--lora_targets", type=str,   default="Wqkv,proj,fc",
                   help="Comma-separated attribute names to wrap with LoRA inside "
                        "core_block. 'proj' matches both attn.proj and mlp.proj.")
    return p.parse_args()


ARGS       = None
DEVICE     = "cuda"
DTYPE      = torch.bfloat16
tokenizer  = None
OUTPUT_DIR = None


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoraLinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta W += B @ A * scale."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_f, out_f = base.in_features, base.out_features
        self.base    = base
        self.lora_A  = nn.Linear(in_f,  rank, bias=False)
        self.lora_B  = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / rank

        # Standard LoRA init: A ~ kaiming uniform, B = 0 → delta starts as no-op.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Cast LoRA weights to match base dtype (bfloat16 on HPC)
        self.lora_A.to(dtype=base.weight.dtype)
        self.lora_B.to(dtype=base.weight.dtype)

        # Freeze base
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def inject_lora(model, rank: int, alpha: float, target_attrs: list[str]) -> int:
    """Replace matching Linear layers inside core_block with LoraLinear wrappers.

    Only modules whose *parent path* includes 'core_block' and whose attribute
    name is in target_attrs are wrapped.  Returns the number of replaced layers.
    """
    replaced = 0
    for mod_name, module in list(model.named_modules()):
        if "core_block" not in mod_name:
            continue
        for attr in target_attrs:
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear):
                setattr(module, attr, LoraLinear(child, rank, alpha))
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Dataset
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
def load_model_with_lora():
    cfg = RavenConfig.from_pretrained(ARGS.model_name)
    cfg.iter_injection      = "none"
    cfg.ccot_injection      = "none"
    cfg.proj_bottleneck_dim = 0

    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name,
        config=cfg,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        ignore_mismatched_sizes=True,
    )

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into core_block
    alpha   = ARGS.lora_alpha if ARGS.lora_alpha else float(ARGS.lora_rank)
    targets = [t.strip() for t in ARGS.lora_targets.split(",")]
    n_replaced = inject_lora(model, rank=ARGS.lora_rank, alpha=alpha,
                             target_attrs=targets)
    log(f"LoRA injected into {n_replaced} layers  (rank={ARGS.lora_rank}, alpha={alpha})")

    # Mark only LoRA parameters as trainable
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True

    if ARGS.grad_checkpoint:
        model.gradient_checkpointing_enable()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    return model


def save_checkpoint(model):
    path = OUTPUT_DIR / "lora"
    path.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    torch.save(lora_state, path / "lora_weights.pt")
    model.config.save_pretrained(str(path))
    size_mb = sum(t.numel() * t.element_size() for t in lora_state.values()) / 1e6
    log(f"Saved → {path}  ({sum(t.numel() for t in lora_state.values()):,} params, {size_mb:.0f} MB)")


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

    # TBPTT: num_steps split into no-grad prefix + backprop suffix
    no_grad  = max(0, ARGS.num_steps - ARGS.backprop_depth)
    num_steps_tensor = torch.tensor([no_grad, ARGS.backprop_depth])

    log(f"Single-pass LoRA training: {ARGS.num_steps} steps "
        f"(no-grad={no_grad}, grad={ARGS.backprop_depth})")
    log(f"Epochs={ARGS.epochs}  steps/epoch={len(loader)}  "
        f"eff_batch={ARGS.batch_size * ARGS.grad_accum}  LR={ARGS.lr}")

    global_step = 0
    accum_loss  = 0.0
    accum_count = 0

    for epoch in range(ARGS.epochs):
        for batch_idx, batch in enumerate(loader):
            iids   = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out  = model(input_ids=iids, labels=labels,
                         num_steps=num_steps_tensor)
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
# Load for eval (merges LoRA weights into model for eval_ccot.py compatibility)
# ---------------------------------------------------------------------------
def load_lora_checkpoint(run_dir: str):
    """Load base model and overlay LoRA weights for downstream evaluation."""
    path = Path(run_dir)
    cfg  = RavenConfig.from_pretrained(str(path))
    cfg.iter_injection = "none"
    cfg.ccot_injection = "none"

    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name, config=cfg, torch_dtype=DTYPE,
        device_map=DEVICE, ignore_mismatched_sizes=True,
    )
    for param in model.parameters():
        param.requires_grad = False

    alpha   = ARGS.lora_alpha if ARGS.lora_alpha else float(ARGS.lora_rank)
    targets = [t.strip() for t in ARGS.lora_targets.split(",")]
    inject_lora(model, rank=ARGS.lora_rank, alpha=alpha, target_attrs=targets)

    weights_path = path / "lora_weights.pt"
    if weights_path.exists():
        delta    = torch.load(str(weights_path), map_location=DEVICE)
        missing, unexpected = model.load_state_dict(delta, strict=False)
        if unexpected:
            log(f"WARNING: unexpected keys: {unexpected}")
        log(f"Loaded {len(delta)} LoRA tensors from {weights_path}")
    else:
        log(f"WARNING: no lora_weights.pt found at {path}")

    model.eval()
    return model


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
    log(f"LoRA rank={ARGS.lora_rank}  targets={ARGS.lora_targets}")

    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(vars(ARGS), indent=2, default=str)
    )

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name)

    log("Loading model with LoRA...")
    model = load_model_with_lora()

    train(model)

    log("Done.")


if __name__ == "__main__":
    main()
