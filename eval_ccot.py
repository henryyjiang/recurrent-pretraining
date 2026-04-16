"""
Evaluate CCoT checkpoints with the lm-eval harness — same benchmarks as the paper.

Loads each checkpoint by overlaying trainable_weights.pt onto the base HF model,
wraps it in lm_eval's HFLM, and runs simple_evaluate.

Usage:
    python eval_ccot.py \\
        --run_dir   ./runs/ccot_20260413_012258 \\
        --tasks     arc_challenge,hellaswag,winogrande,mmlu,gsm8k \\
        --num_steps 8,16,32 \\
        --output_dir ./runs/ccot_20260413_012258/eval \\
        --model_name tomg-group-umd/huginn-0125 \\
        --hf_cache  ~/scratch/hf_cache
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

# ── Unbuffered stdout for .out files ──────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Bootstrap recpre without triggering __init__.py ───────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

for _name, _rel in [
    ("recpre.raven_config_minimal",   "recpre/raven_config_minimal.py"),
    ("recpre.raven_modeling_minimal", "recpre/raven_modeling_minimal.py"),
]:
    _path = REPO_ROOT / _rel
    _spec = importlib.util.spec_from_file_location(_name, str(_path))
    _mod  = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recpre.raven_config_minimal import RavenConfig

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


DTYPE  = torch.bfloat16
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Multi-pass wrapper
# ---------------------------------------------------------------------------
class MultiPassWrapper(torch.nn.Module):
    """Wraps RavenForCausalLM to perform N-1 no-grad warmup passes before any
    scored forward or generate call.

    Each warmup pass re-reads the same input tokens and chains the final loop
    latent state (ccot_memory) into the next pass via ccot_proj.  This mirrors
    the latent-CoT training procedure in train_ccot.py: the model refines an
    internal "scratchpad" state over multiple passes before committing to a
    prediction, analogous to chain-of-thought token generation but without
    materialising intermediate tokens.

    ccot_memory resets to None for every new batch — there is no cross-example
    chaining during evaluation.
    """

    def __init__(self, base_model: torch.nn.Module,
                 num_passes: int, steps_per_pass: int):
        super().__init__()
        self.base_model    = base_model
        self.num_passes    = num_passes
        self.steps_per_pass = steps_per_pass
        # Proxy config so HFLM can introspect model_type, vocab_size, etc.
        self.config = base_model.config

    # ── Attribute proxy (forwards unknown attrs to base_model) ───────────
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    # ── Warmup ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _warmup(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Run num_passes-1 warmup passes; return the final ccot_memory."""
        if self.num_passes <= 1:
            return None
        od = {"return_logits": False, "return_latents": True,
              "return_head":   False, "return_stats":   False}
        ccot_memory = None
        for _ in range(self.num_passes - 1):
            out = self.base_model(
                input_ids=input_ids,
                num_steps=self.steps_per_pass,
                ccot_memory=ccot_memory,
                output_details=od,
            )
            ccot_memory = out.latent_states.detach()
        return ccot_memory

    # ── Scored forward (loglikelihood tasks) ─────────────────────────────
    def forward(self, input_ids: torch.Tensor, **kwargs):
        ccot_memory = self._warmup(input_ids)
        kwargs["num_steps"] = self.steps_per_pass
        return self.base_model(input_ids=input_ids,
                               ccot_memory=ccot_memory, **kwargs)

    # ── Generation tasks ─────────────────────────────────────────────────
    def generate(self, input_ids: torch.Tensor, **kwargs):
        ccot_memory = self._warmup(input_ids)
        kwargs["num_steps"] = self.steps_per_pass
        return self.base_model.generate(input_ids=input_ids,
                                        ccot_memory=ccot_memory, **kwargs)


# ── Checkpoint definitions ─────────────────────────────────────────────────
# Each entry: (label, iter_injection, ccot_injection, subdir_or_None)
# subdir_or_None=None means the frozen base model (no delta loaded).
EXPERIMENTS = [
    ("Baseline",  "none", "none", None),
    ("iter_only", "add",  "none", "iter_only"),
    ("ccot_only", "none", "add",  "ccot_only"),
    ("both",      "add",  "add",  "both"),
]


def load_model(model_name, run_dir, subdir, iter_injection, ccot_injection,
               num_passes: int = 1, steps_per_pass: Optional[int] = None):
    cfg = RavenConfig.from_pretrained(model_name)
    cfg.iter_injection = iter_injection
    cfg.ccot_injection = ccot_injection

    log(f"  Loading base model from HF cache...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        ignore_mismatched_sizes=True,
    )

    if subdir is not None:
        weights_path = run_dir / subdir / "trainable_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {weights_path}")
        delta = torch.load(str(weights_path), map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(delta, strict=False)
        if unexpected:
            log(f"  WARNING: unexpected keys: {unexpected}")
        log(f"  Overlaid {len(delta)} tensors from {weights_path}")

    model.eval()

    # Wrap with MultiPassWrapper for CCoT experiments so that lm-eval sees a
    # standard model interface while the warmup passes are handled transparently.
    if ccot_injection != "none" and num_passes > 1 and steps_per_pass is not None:
        log(f"  Wrapping with MultiPassWrapper "
            f"(num_passes={num_passes}, steps_per_pass={steps_per_pass})")
        model = MultiPassWrapper(model, num_passes=num_passes,
                                 steps_per_pass=steps_per_pass)

    return model


def run_eval(model, tokenizer, tasks, num_steps, num_fewshot, batch_size, limit):
    """Wrap model in HFLM and run lm-eval simple_evaluate."""
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        # Pass num_steps to every model.generate() call
        add_bos_token=False,
    )

    # gen_kwargs forwards kwargs to model.generate()
    gen_kwargs = {"num_steps": num_steps}

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        gen_kwargs=gen_kwargs,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",    required=True,
                        help="Path to the run directory containing checkpoints")
    parser.add_argument("--model_name", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--hf_cache",   default=None)
    parser.add_argument("--tasks",      default="arc_challenge,hellaswag,winogrande,mmlu,gsm8k",
                        help="Comma-separated lm-eval task names")
    parser.add_argument("--num_steps",  default="8,16,32",
                        help="Comma-separated recurrence depths to sweep")
    parser.add_argument("--num_fewshot",type=int, default=None,
                        help="Few-shot examples. None = task default (e.g. 8 for GSM8K)")
    parser.add_argument("--batch_size", default="auto",
                        help="lm-eval batch size")
    parser.add_argument("--limit",      type=float, default=None,
                        help="Fraction or count of examples to evaluate (None = full)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save JSON results (default: <run_dir>/eval)")
    parser.add_argument("--experiments", nargs="+",
                        default=["Baseline", "iter_only", "ccot_only", "both"],
                        help="Which experiments to evaluate")
    parser.add_argument("--num_passes",  type=int, default=4,
                        help="Number of full-transformer passes per example for CCoT "
                             "experiments (must match the value used during training). "
                             "steps_per_pass = num_steps // num_passes")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"]            = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
        os.environ["HF_DATASETS_CACHE"]  = args.hf_cache

    run_dir    = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks    = args.tasks.split(",")
    steps    = [int(s) for s in args.num_steps.split(",")]

    log(f"Run dir:    {run_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Tasks:      {tasks}")
    log(f"Steps:      {steps}")
    log(f"Experiments: {args.experiments}")
    log(f"num_passes: {args.num_passes}")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_results = {}

    for label, iter_inj, ccot_inj, subdir in EXPERIMENTS:
        if label not in args.experiments:
            continue

        log(f"\n{'═'*60}")
        log(f"  {label}  (iter={iter_inj!r}, ccot={ccot_inj!r})")
        log(f"{'═'*60}")

        all_results[label] = {}

        for num_steps in steps:
            # For CCoT experiments, wrap the model so each lm-eval call triggers
            # num_passes-1 warmup passes internally.  steps_per_pass is derived
            # from num_steps so that the total recurrent budget matches the
            # eval_steps sweep (e.g. num_steps=32, num_passes=4 → 8 steps/pass).
            steps_per_pass = (max(1, num_steps // args.num_passes)
                              if ccot_inj != "none" and args.num_passes > 1
                              else None)
            model = load_model(
                args.model_name, run_dir, subdir, iter_inj, ccot_inj,
                num_passes=args.num_passes,
                steps_per_pass=steps_per_pass,
            )

            log(f"\n  → num_steps={num_steps}"
                + (f" ({args.num_passes}×{steps_per_pass})" if steps_per_pass else ""))
            results = run_eval(
                model, tokenizer, tasks,
                num_steps=steps_per_pass if steps_per_pass else num_steps,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                limit=args.limit,
            )

            # Print summary
            for task in tasks:
                if task in results["results"]:
                    r = results["results"][task]
                    # GSM8K uses exact_match, others use acc or acc_norm
                    metric = (
                        r.get("exact_match,flexible-extract")
                        or r.get("exact_match,strict-match")
                        or r.get("acc_norm,none")
                        or r.get("acc,none")
                    )
                    if metric is not None:
                        log(f"    {task:<25} {metric*100:.1f}%")

            all_results[label][num_steps] = results["results"]

            # Save per-checkpoint-per-steps JSON
            out_path = output_dir / f"{label}_steps{num_steps}.json"
            out_path.write_text(
                json.dumps(results["results"], indent=2, default=str)
            )
            log(f"  Saved → {out_path}")

            del model
            torch.cuda.empty_cache()

    # Save combined summary
    summary_path = output_dir / "all_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    log(f"\nAll results → {summary_path}")

    # Print comparison table
    log(f"\n{'='*70}")
    log("SUMMARY — GSM8K exact match")
    log(f"{'='*70}")
    log(f"{'Model':<14}" + "".join(f"{'steps='+str(s):>12}" for s in steps))
    log("-" * (14 + 12 * len(steps)))
    for label in args.experiments:
        if label not in all_results:
            continue
        row = f"{label:<14}"
        for s in steps:
            r = all_results[label].get(s, {})
            gsm = r.get("gsm8k", {})
            metric = (
                gsm.get("exact_match,flexible-extract")
                or gsm.get("exact_match,strict-match")
            )
            row += f"{'N/A':>12}" if metric is None else f"{metric*100:>11.1f}%"
        log(row)


if __name__ == "__main__":
    main()
