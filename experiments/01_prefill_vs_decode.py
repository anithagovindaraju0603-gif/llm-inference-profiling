import os
import time
import torch
import statistics
import pandas as pd
from setup import model_download

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "01_prefill_vs_decode.csv")

model, tokenizer, device = model_download.load_model()

PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]
results = []

# ── Helper: build input of exact token length ─────────────────────────────────
# Tokenization is done OUTSIDE the timing window — we're measuring inference,
# not tokenization. Truncation ensures we hit exactly the target length.
def make_input(length):
    prompt = "hello " * length
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=length
    ).to(device)
    return tokens

# ── Warmup ────────────────────────────────────────────────────────────────────
# First runs include CUDA kernel compilation and cache warming — not representative. 3 times!!
# We use model() directly (not generate) to match what prefill measurement does.
dummy = make_input(512)
for _ in range(3):
    with torch.no_grad():
        model(**dummy)
    torch.cuda.synchronize()  # GPU ops are async — force wait before continuing

# ── Prefill + Decode measurement ─────────────────────────────────────────────
# Combined loop: prefill is timed first, KV cache is captured and immediately
# reused for the decode step. Cache is freed at end of each iteration.
for length in PROMPT_LENGTHS:
    inputs = make_input(length)
    actual_length = inputs["input_ids"].shape[1]

    times = []
    for _ in range(5):
        # prefill — timed, use_cache=True captures KV cache for decode reuse
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            prefill_out = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    prefill_elapsed = statistics.median(times)
    prefill_mem_gb = torch.cuda.memory_allocated() / 1e9 
    past_key_values = prefill_out.past_key_values

    results.append({
        "phase": "prefill",
        "prompt_length": actual_length,
        "time_sec": prefill_elapsed,
        "tokens_per_sec": actual_length / prefill_elapsed,
        "mem_allocated_gb": prefill_mem_gb
    })
    print(f"prefill | length={actual_length} | {prefill_elapsed:.4f}s | {actual_length / prefill_elapsed:.1f} tok/s")

    # decode — timed, feeds last token with pre-built KV cache
    next_token = inputs["input_ids"][:, -1:]

    decode_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            model(next_token, past_key_values=past_key_values, use_cache=True)

        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - start)

    decode_elapsed = statistics.median(decode_times)
    decode_mem_gb = torch.cuda.memory_allocated() / 1e9 

    results.append({
        "phase": "decode",
        "prompt_length": actual_length,
        "time_sec": decode_elapsed,
        "tokens_per_sec": 1 / decode_elapsed,
        "mem_allocated_gb": decode_mem_gb
    })
    print(f"decode  | kv_length={actual_length} | {decode_elapsed:.4f}s | {1 / decode_elapsed:.1f} tok/s")

# ── Save results ──────────────────────────────────────────────────────────────
# Save as CSV so the analysis notebook can load it without re-running.
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
df.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved to {RESULTS_PATH}")
