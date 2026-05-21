# experiments/02_per_layer_profile.py
#
# Goal: Break down a single forward pass into per-operation CUDA time.
# Produces:
#   - results/02_decode_profile_table.csv   (operation-level timing)
#   - results/02_prefill_profile_table.csv
#   - results/traces/decode_trace.json      (open in chrome://tracing)
#   - results/traces/prefill_trace.json

import os
import torch
import pandas as pd
from torch.profiler import profile, ProfilerActivity, record_function
from setup import model_download

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TRACE_DIR   = os.path.join(RESULTS_DIR, "traces")
os.makedirs(TRACE_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
model, tokenizer, device = model_download.load_model()

# ── Helper ────────────────────────────────────────────────────────────────────
def make_input(length):
    prompt = "hello " * length
    return tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=length
    ).to(device)

# ── Warmup — same discipline as Experiment 1 ─────────────────────────────────
dummy = make_input(512)
for _ in range(3):
    with torch.no_grad():
        model(**dummy)
torch.cuda.synchronize()

print("Warmup done. Starting profiling runs...")

# ─────────────────────────────────────────────────────────────────────────────
# PROFILING SESSION 1: Decode step at KV length = 512
# This is the bread-and-butter inference case we care most about.
# ─────────────────────────────────────────────────────────────────────────────
inputs = make_input(512)

# Build KV cache from prefill first (outside the profiling window)
with torch.no_grad():
    prefill_out = model(**inputs, use_cache=True)
past_kv = prefill_out.past_key_values
next_token = inputs["input_ids"][:, -1:]
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,   # True gives call stacks but makes traces huge; off for now
) as decode_prof:
    for _ in range(5):  # profile 5 decode steps, profiler averages them
        with record_function("decode_step"):
            with torch.no_grad():
                model(next_token, past_key_values=past_kv, use_cache=True)

torch.cuda.synchronize()

# Save Chrome trace — open this in chrome://tracing or https://ui.perfetto.dev
decode_prof.export_chrome_trace(os.path.join(TRACE_DIR, "decode_trace.json"))

# Build the summary table
decode_table = decode_prof.key_averages()
print("\n=== DECODE STEP — Top operations by CUDA time ===")
print(decode_table.table(sort_by="cuda_time_total", row_limit=30))

# Parse into a DataFrame for CSV export
rows = []
for evt in decode_table:
    rows.append({
        "name": evt.key,
        "cuda_time_ms": evt.cuda_time_total / 1000,   # microseconds → ms
        "cpu_time_ms":  evt.cpu_time_total  / 1000,
        "cuda_pct":     evt.cuda_time_total / max(sum(e.cuda_time_total for e in decode_table), 1) * 100,
        "calls":        evt.count,
        "input_shapes": str(evt.input_shapes) if hasattr(evt, "input_shapes") else "",
    })

df_decode = pd.DataFrame(rows).sort_values("cuda_time_ms", ascending=False)
df_decode.to_csv(os.path.join(RESULTS_DIR, "02_decode_profile_table.csv"), index=False)
print(f"\nSaved decode table → results/02_decode_profile_table.csv")
print(f"Saved decode trace → results/traces/decode_trace.json")


# ─────────────────────────────────────────────────────────────────────────────
# PROFILING SESSION 2: Prefill at 512 tokens
# Run separately so the profiler windows don't overlap.
# ─────────────────────────────────────────────────────────────────────────────
inputs_fresh = make_input(512)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prefill_prof:
    for _ in range(3):   # fewer reps — prefill is slow
        with record_function("prefill_step"):
            with torch.no_grad():
                model(**inputs_fresh, use_cache=True)

torch.cuda.synchronize()

prefill_prof.export_chrome_trace(os.path.join(TRACE_DIR, "prefill_trace.json"))

prefill_table = prefill_prof.key_averages()
print("\n=== PREFILL STEP — Top operations by CUDA time ===")
print(prefill_table.table(sort_by="cuda_time_total", row_limit=30))

rows = []
for evt in prefill_table:
    rows.append({
        "name": evt.key,
        "cuda_time_ms": evt.cuda_time_total / 1000,
        "cpu_time_ms":  evt.cpu_time_total  / 1000,
        "cuda_pct":     evt.cuda_time_total / max(sum(e.cuda_time_total for e in prefill_table), 1) * 100,
        "calls":        evt.count,
        "input_shapes": str(evt.input_shapes) if hasattr(evt, "input_shapes") else "",
    })

df_prefill = pd.DataFrame(rows).sort_values("cuda_time_ms", ascending=False)
df_prefill.to_csv(os.path.join(RESULTS_DIR, "02_prefill_profile_table.csv"), index=False)
print(f"\nSaved prefill table → results/02_prefill_profile_table.csv")
print(f"Saved prefill trace → results/traces/prefill_trace.json")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Quick side-by-side summary — the comparison you'll want for the blog
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SIDE-BY-SIDE: Decode vs Prefill — Top 10 ops ===")
top_decode  = df_decode.head(10)[["name", "cuda_time_ms", "cuda_pct"]].rename(
    columns={"cuda_time_ms": "decode_ms", "cuda_pct": "decode_pct"})
top_prefill = df_prefill.head(10)[["name", "cuda_time_ms", "cuda_pct"]].rename(
    columns={"cuda_time_ms": "prefill_ms", "cuda_pct": "prefill_pct"})
comparison = top_decode.merge(top_prefill, on="name", how="outer").fillna(0)
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_DIR, "02_comparison.csv"), index=False)