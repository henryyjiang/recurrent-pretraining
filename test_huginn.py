"""
Quick test script for Huginn (tomg-group-umd/huginn-0125).
Run with: /opt/miniconda3/envs/huginn/bin/python3 test_huginn.py

Tests:
  1. Basic sanity — does it generate coherent text?
  2. Recurrence depth sweep — does more thinking help?
  3. Latent convergence — how fast does the loop state stabilize?
  4. Harder reasoning — multi-step math and logic
  5. CCoT episodic memory — latent_states from query N passed as ccot_memory to query N+1

Config flags (set on the RavenConfig before loading):
  iter_injection : "none" | "add" | "prepend"
      Within-query, iteration-to-iteration injection via iter_proj.
  ccot_injection : "none" | "add" | "prepend"
      Cross-query episodic memory via ccot_proj (pass ccot_memory to forward()).
"""

import sys
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load the two minimal modeling files WITHOUT triggering
# recpre/__init__.py (which imports recpre.utils, which references a torch
# internal symbol removed in torch 2.11).
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
for _name, _path in [
    ("recpre.raven_config_minimal", "recpre/raven_config_minimal.py"),
    ("recpre.raven_modeling_minimal", "recpre/raven_modeling_minimal.py"),
]:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
MODEL_NAME = "tomg-group-umd/huginn-0125"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

print(f"Device: {DEVICE} | dtype: {DTYPE}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (uses cached weights after first download)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
model.eval()
print(f"Loaded {type(model).__name__} — "
      f"prelude={model.config.n_layers_in_prelude}, "
      f"loop_layers={model.config.n_layers_in_recurrent_block}, "
      f"coda={model.config.n_layers_in_coda}, "
      f"mean_recurrence={model.config.mean_recurrence}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenize(text: str):
    """Returns input_ids and attention_mask on the right device.
    Drops token_type_ids — the tokenizer adds it but Huginn doesn't accept it.
    """
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


@torch.no_grad()
def generate(prompt: str, num_steps: int = 32, max_new_tokens: int = 64) -> str:
    input_ids, attn_mask = tokenize(prompt)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        num_steps=num_steps,
        do_sample=False,
    )
    # Decode only the newly generated tokens
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def get_latent_trajectory(prompt: str, num_steps: int = 32):
    """Run the loop manually and record the norm of state change each step."""
    input_ids, _ = tokenize(prompt)
    input_ids = input_ids

    # Embed + prelude
    freqs_cis = model.freqs_cis[:, :input_ids.shape[1]]
    x = model.transformer.wte(input_ids) * model.emb_scale
    block_idx = torch.tensor(-1, device="cpu", dtype=torch.long)
    for block in model.transformer.prelude:
        block_idx += 1
        x = block(x, freqs_cis, block_idx, None, None)
    input_embeds = x  # z_0

    # Initialize loop state
    state = model.initialize_state(input_embeds)
    deltas = []
    block_idx = torch.tensor(1, device="cpu", dtype=torch.long)

    for step in range(num_steps):
        prev = state.clone()
        state, block_idx = model.core_block_forward(
            state, input_embeds, freqs_cis, None, None,
            block_idx=torch.tensor(1, device="cpu", dtype=torch.long),
            current_step=step,
        )
        delta = (state - prev).norm(dim=-1).mean().item()
        deltas.append(delta)

    return deltas


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Test 1: Basic sanity
# ---------------------------------------------------------------------------
section("TEST 1: Basic sanity")
prompts = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Once upon a time there was a",
]
for p in prompts:
    out = generate(p, num_steps=32, max_new_tokens=30)
    print(f"  PROMPT: {repr(p)}")
    print(f"  OUTPUT: {repr(out)}\n")


# ---------------------------------------------------------------------------
# Test 2: Recurrence depth sweep
# ---------------------------------------------------------------------------
section("TEST 2: Recurrence depth sweep (does more thinking help?)")
math_prompts = [
    ("Simple",   "What is 8 × 7? Answer:"),
    ("Medium",   "If a train travels at 60 mph for 2.5 hours, how far does it go? Answer:"),
    ("Multi-step","Sarah has 24 apples. She gives half to Tom, then buys 6 more. How many does she have? Answer:"),
]
for label, prompt in math_prompts:
    print(f"  [{label}] {prompt}")
    for steps in [1, 4, 8, 16, 32]:
        out = generate(prompt, num_steps=steps, max_new_tokens=20)
        print(f"    steps={steps:2d}: {repr(out.strip())}")
    print()


# ---------------------------------------------------------------------------
# Test 3: Latent convergence trajectory
# ---------------------------------------------------------------------------
section("TEST 3: Latent convergence — state delta per loop step")
traj_prompts = [
    ("Easy:  2+2",          "What is 2 + 2? Answer:"),
    ("Hard:  multi-step",   "A store sells apples for $0.50 each and oranges for $0.75 each. "
                            "If I buy 4 apples and 3 oranges, how much do I spend in total? Answer:"),
]
for label, prompt in traj_prompts:
    deltas = get_latent_trajectory(prompt, num_steps=32)
    print(f"  {label}")
    # Print every 4th step to keep output compact
    for i, d in enumerate(deltas):
        if i % 4 == 0 or i == len(deltas) - 1:
            bar = "█" * int(d * 2)
            print(f"    step {i+1:2d}: delta={d:.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Test 4: Harder reasoning
# ---------------------------------------------------------------------------
section("TEST 4: Harder reasoning (steps=32 vs steps=4)")
hard_prompts = [
    "Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nA:",
    "Q: If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?\nA:",
    "Q: In a lake, there is a patch of lily pads. Every day the patch doubles in size. "
    "If it takes 48 days to cover the whole lake, how long does it take to cover half the lake?\nA:",
]
for prompt in hard_prompts:
    print(f"  PROMPT: {prompt.splitlines()[0]}")
    for steps in [4, 32]:
        out = generate(prompt, num_steps=steps, max_new_tokens=30)
        print(f"    steps={steps:2d}: {repr(out.strip())}")
    print()

# ---------------------------------------------------------------------------
# Test 5: Cross-query CCoT episodic memory
# ---------------------------------------------------------------------------
section("TEST 5: Cross-query CCoT episodic memory (latent_states passthrough)")
# Demonstrate the API: run Q1, capture latent_states, pass as ccot_memory to Q2.
# With ccot_injection="none" (default) or zero-init ccot_proj, outputs should be
# identical whether ccot_memory is provided or not — confirms the no-op baseline.
# After finetuning with ccot_injection="add" or "prepend", Q2 should change.

@torch.no_grad()
def forward_and_capture(prompt: str, num_steps: int = 8, ccot_memory=None):
    """Single forward pass — returns (generated_text, latent_states)."""
    input_ids, attn_mask = tokenize(prompt)
    output = model(
        input_ids=input_ids,
        num_steps=torch.tensor([0, num_steps]),  # all steps with grad disabled at test time
        ccot_memory=ccot_memory,
        output_details={"return_logits": True, "return_latents": True,
                        "return_head": False, "return_stats": False},
    )
    next_tok = output.logits[0, -1].argmax()
    text = tokenizer.decode(next_tok, skip_special_tokens=True)
    return text, output.latent_states  # latent_states: [B, S, E]

q1 = "The speed of light in a vacuum is approximately"
q2 = "Therefore, one light-year is approximately"

print(f"  Q1: {repr(q1)}")
tok1, mem1 = forward_and_capture(q1, num_steps=8)
print(f"  Q1 next-token: {repr(tok1)}")
print(f"  latent_states shape: {mem1.shape}  (this is ccot_memory for Q2)\n")

print(f"  Q2 (no memory):   ", end="")
tok2_no_mem, _ = forward_and_capture(q2, num_steps=8, ccot_memory=None)
print(repr(tok2_no_mem))

print(f"  Q2 (with Q1 mem): ", end="")
tok2_with_mem, _ = forward_and_capture(q2, num_steps=8, ccot_memory=mem1)
print(repr(tok2_with_mem))

print("\n  NOTE: outputs are identical pre-finetuning because ccot_proj is zero-init.")
print("  After finetuning with ccot_injection='add' or 'prepend', Q2 should diverge.")

print("\nAll tests complete.")
