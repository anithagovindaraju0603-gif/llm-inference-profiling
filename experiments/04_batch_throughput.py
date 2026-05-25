import os
import gc
import time
import torch
import statistics
import pandas as pd
from setup import model_download

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "04_batch_throughput.csv")

model, tokenizer, device = model_download.load_model()

# ── Constants ────────────────────────────────────────────────────────────────────
prompt_len = 512
batch_size = [1, 2, 4, 8, 16, 32]
decode_steps = 50

# ── Helper ────────────────────────────────────────────────────────────────────
def make_input(length, batch_size=1):
    prompt = "hello " * length
    tokens = tokenizer(prompt, return_tensors="pt", truncation = True, max_length=length).to(device)
    if batch_size > 1:
        tokens["input_ids"] = tokens["input_ids"].expand(batch_size, -1)
        tokens["attention_mask"] = tokens["attention_mask"].expand(batch_size, -1)
    return tokens

def mem_gb():
    """Current allocated GPU memory in GB."""
    return torch.cuda.memory_allocated() / 1e9

# ── Warmup ────────────────────────────────────────────────────────────────────
# First runs include CUDA kernel compilation and cache warming — not representative. 3 times!!
# We use model() directly (not generate) to match what prefill measurement does.
dummy = make_input(512)
for _ in range(3):
    with torch.no_grad():
        model(**dummy)
torch.cuda.synchronize()

print("Warmup done. Starting profiling runs...")

# ── Main loop: batch size sweep at fixed prompt length and decode steps ─────────────────────────────────────────────
results = []
for batch in batch_size:
    torch.cuda.empty_cache()
    gc.collect()

    #prefill
    try:
        inputs = make_input(prompt_len, batch_size=batch)
        with torch.no_grad():
            prefill_out = model(**inputs, use_cache=True)
            past_kv = prefill_out.past_key_values
            del prefill_out
        torch.cuda.synchronize()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch size {batch} → OOM during prefill")
            results.append({
                "batch_size": batch,
                "prompt_len": prompt_len,
                "p50_ms": None,
                "p99_ms": None,
                "tps_per_request": None,
                "total_throughput_tps": None,
                "mem_allocated_gb": None,
                "status": "OOM"
            })
            torch.cuda.empty_cache()
            gc.collect()
            continue
    except:
        print(f"Batch size {batch} → Unexpected error during prefill")
        raise

    #decode
    step_times = []
    next_token = inputs["input_ids"][:,-1:]
    try:
        for i in range(decode_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                decode_out = model(next_token, past_key_values=past_kv, use_cache=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)
            next_token = decode_out.logits[:, -1:].argmax(-1)
            past_kv = decode_out.past_key_values
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch size {batch} → OOM during decode")
            results.append({"batch_size": batch,
                "prompt_len": prompt_len,
                "p50_ms": None,
                "p99_ms": None,
                "tps_per_request": None,
                "total_throughput_tps": None,
                "mem_allocated_gb": None,
                "status": "OOM_decode"})
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            raise
    #Compute stats - p50 and p99
    p50 = statistics.median(step_times)
    p99 = statistics.quantiles(step_times, n=100)[98] if len(step_times) >= 100 else max(step_times)  # 99th percentile is the 98th quantile in 100 divisions
    print(f"Batch size {batch} → p50 decode step time: {p50:.3f}s, p99 decode step time: {p99:.3f}s")
    tps_per_request = 1 / p50 # tokens per second per request
    total_throughput = batch * tps_per_request
    print(f"Batch size {batch} → Total throughput: {total_throughput:.1f} tokens/s")

    #Capture memory utilization during decode steps
    mem_util = mem_gb()

    del past_kv, decode_out
    torch.cuda.empty_cache()
    gc.collect()

    # inside the loop, after computing stats:
    results.append({
        "batch_size": batch,
        "prompt_len": prompt_len,
        "p50_ms": round(p50 * 1000, 2),
        "p99_ms": round(p99 * 1000, 2),
        "tps_per_request": round(tps_per_request, 1),
        "total_throughput_tps": round(total_throughput, 1),
        "mem_allocated_gb": round(mem_util, 3),
        "status": "ok"
    })

pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
print(f"\nSaved results → {RESULTS_PATH}")





