# experiments/03_kv_cache_memory.py
#
# Goal: Measure KV cache memory growth and verify against theoretical prediction.
# Produces:
#   results/03_kv_cache_snapshots.csv   — memory at each prefill length + decode steps
#   results/03_oom_boundary.csv         — batch size OOM sweep
#   results/03_theory_vs_actual.csv     — predicted vs measured per seq_len

import os
import gc
import torch
import pandas as pd
from setup import model_download

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

model, tokenizer, device = model_download.load_model()

# ── Helper ────────────────────────────────────────────────────────────────────

def make_input(length, batch_size=1):
    prompt = "hello " * length
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=length
    ).to(device)
    if batch_size > 1:
        tokens["input_ids"] = tokens["input_ids"].expand(batch_size, -1)
        tokens["attention_mask"] = tokens["attention_mask"].expand(batch_size, -1)
    return tokens

def mem_gb():
    """Current allocated GPU memory in GB."""
    return torch.cuda.memory_allocated() / 1e9

def reserved_gb():
    """Current reserved GPU memory in GB (PyTorch allocator view)."""
    return torch.cuda.memory_reserved() / 1e9

# ─────────────────────────────────────────────────────────────────────────────
# SUB-TASK A: Theoretical KV cache prediction
# Formula: 2 (K+V) × num_kv_heads × head_dim × num_layers × seq_len × batch × dtype_bytes
# LLaMA-3-8B: 32 layers, 8 KV heads (GQA!), 128 head_dim, FP16 = 2 bytes
# ─────────────────────────────────────────────────────────────────────────────

cfg = model.config
num_layers    = cfg.num_hidden_layers          # 32
num_kv_heads  = cfg.num_key_value_heads        # 8  ← GQA, NOT 32
head_dim      = cfg.hidden_size // cfg.num_attention_heads  # 128
dtype_bytes   = 2  # FP16

print("=== LLaMA-3-8B KV cache config ===")
print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print()

theory_rows = []
for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    kv_bytes = 2 * num_kv_heads * head_dim * num_layers * seq_len * 1 * dtype_bytes
    kv_gb    = kv_bytes / 1e9
    theory_rows.append({"seq_len": seq_len, "predicted_kv_cache_gb": round(kv_gb, 4)})
    print(f"  seq_len={seq_len:6d} → predicted KV cache = {kv_gb:.3f} GB")

df_theory = pd.DataFrame(theory_rows)

# ─────────────────────────────────────────────────────────────────────────────
# SUB-TASK B: Measure actual memory growth
# Capture: baseline (weights only), after prefill, per decode step
# ─────────────────────────────────────────────────────────────────────────────

# Warmup
dummy = make_input(512)
for _ in range(3):
    with torch.no_grad():
        model(**dummy)
torch.cuda.synchronize()

baseline_gb = mem_gb()
print(f"\nModel weights baseline: {baseline_gb:.2f} GB")

PREFILL_LENGTHS = [512, 1024, 2048, 4096, 8192]
DECODE_STEPS    = 50
SNAPSHOT_EVERY  = 10

snapshot_rows = []
theory_actual_rows = []

for length in PREFILL_LENGTHS:
    print(f"\n--- Prefill length={length} ---")
    inputs = make_input(length)

    # Measure memory right after prefill
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    torch.cuda.synchronize()

    past_kv = out.past_key_values  # save KV cache first
    del out                         # delete everything else
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_after_prefill = mem_gb()
    kv_cache_overhead = mem_after_prefill - baseline_gb

    print(f"  After prefill:  {mem_after_prefill:.3f} GB total, {kv_cache_overhead:.3f} GB KV cache")

    snapshot_rows.append({
        "event": "prefill",
        "seq_len": length,
        "decode_step": 0,
        "mem_allocated_gb": mem_after_prefill,
        "kv_cache_overhead_gb": kv_cache_overhead,
    })

    # Compare to theory
    predicted = df_theory.loc[df_theory.seq_len == length, "predicted_kv_cache_gb"]
    pred_val  = predicted.values[0] if len(predicted) > 0 else None
    theory_actual_rows.append({
        "seq_len": length,
        "predicted_kv_gb": pred_val,
        "actual_kv_gb": round(kv_cache_overhead, 4),
        "error_pct": round(abs(kv_cache_overhead - pred_val) / pred_val * 100, 2) if pred_val else None,
    })

    # Decode snapshot — watch memory grow token by token
    next_token = inputs["input_ids"][:, -1:]
    for step in range(1, DECODE_STEPS + 1):
        with torch.no_grad():
            step_out = model(next_token, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
        next_token = step_out.logits[:, -1:].argmax(-1)
        past_kv    = step_out.past_key_values

        if step % SNAPSHOT_EVERY == 0:
            m = mem_gb()
            snapshot_rows.append({
                "event": "decode",
                "seq_len": length,
                "decode_step": step,
                "mem_allocated_gb": m,
                "kv_cache_overhead_gb": m - baseline_gb,
            })
            print(f"  decode step {step:3d}: {m:.3f} GB")

    # Free the KV cache before next iteration
    del past_kv, out, step_out
    torch.cuda.empty_cache()
    gc.collect()

df_snapshots = pd.DataFrame(snapshot_rows)
df_theory_actual = pd.DataFrame(theory_actual_rows)

df_snapshots.to_csv(os.path.join(RESULTS_DIR, "03_kv_cache_snapshots.csv"), index=False)
df_theory_actual.to_csv(os.path.join(RESULTS_DIR, "03_theory_vs_actual.csv"), index=False)
print("\n\n=== Theory vs Actual ===")
print(df_theory_actual.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SUB-TASK C: OOM boundary sweep across batch sizes
# Fixed prompt: 2048 tokens. Try batch 1, 2, 4, 8, 16.
# Wrap each in try/except to catch CUDA OOM gracefully.
# ─────────────────────────────────────────────────────────────────────────────

FIXED_PROMPT_LEN = 2048
BATCH_SIZES      = [1, 2, 4, 8, 16]

oom_rows = []
print("\n=== OOM boundary sweep (prompt_len=2048) ===")

for bs in BATCH_SIZES:
    torch.cuda.empty_cache()
    gc.collect()

    try:
        inputs = make_input(FIXED_PROMPT_LEN, batch_size=bs)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()

        m = mem_gb()
        kv_overhead = m - baseline_gb
        status = "ok"
        print(f"  batch_size={bs:2d}: {m:.2f} GB total, {kv_overhead:.2f} GB KV — OK")

        del out, inputs
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            m       = None
            kv_overhead = None
            status  = "OOM"
            print(f"  batch_size={bs:2d}: OOM")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise

    oom_rows.append({
        "batch_size": bs,
        "prompt_len": FIXED_PROMPT_LEN,
        "mem_allocated_gb": m,
        "kv_cache_overhead_gb": kv_overhead,
        "status": status,
    })

df_oom = pd.DataFrame(oom_rows)
df_oom.to_csv(os.path.join(RESULTS_DIR, "03_oom_boundary.csv"), index=False)
print(f"\nSaved → results/03_kv_cache_snapshots.csv")
print(f"Saved → results/03_theory_vs_actual.csv")
print(f"Saved → results/03_oom_boundary.csv")