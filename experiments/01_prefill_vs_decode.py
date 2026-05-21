import os
import time
import torch
import statistics
import nvidia_ml_py as pynvml 
import threading
import pandas as pd
from setup import model_download

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "01_prefill_vs_decode.csv")

model, tokenizer, device = model_download.load_model()
PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]
results = []

# ── GPU utilization sampler ───────────────────────────────────────────────────
# Samples SM utilization % in a background thread while the GPU is working.
# Returns the median — same idea as taking median of timing runs.
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
 
def sample_utilization_during(fn):
    """Run fn(), sample GPU util every 1ms in background, return median %."""
    compute_samples = []
    memory_samples = []
    stop = threading.Event()
 
    def sampler():
        while not stop.is_set():
            rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            compute_samples.append(rates.gpu)
            memory_samples.append(rates.memory)
            time.sleep(0.001)
 
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    fn()
    torch.cuda.synchronize()
    stop.set()
    t.join()
    compute_median = statistics.median(compute_samples) if compute_samples else 0
    memory_median = statistics.median(memory_samples) if memory_samples else 0
    return compute_median, memory_median

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
    past_key_values = prefill_out.past_key_values

    def run_prefill():
        with torch.no_grad():
            model(**inputs, use_cache=True)
 
    prefill_compute, prefill_memory = sample_utilization_during(run_prefill)
    prefill_mem_gb = torch.cuda.memory_allocated() / 1e9 

    results.append({
        "phase": "prefill",
        "prompt_length": actual_length,
        "time_sec": prefill_elapsed,
        "tokens_per_sec": actual_length / prefill_elapsed,
        "mem_allocated_gb": prefill_mem_gb,
        "gpu_memory_util_pct": prefill_memory
    })
    print(f"prefill | length={actual_length} | time_taken_sec={prefill_elapsed:.4f}s | tokens_per_sec={actual_length / prefill_elapsed:.1f} tok/s | gpu_memory={prefill_memory:.1f}% gpu memory util")

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

    def run_decode_loop():
        for _ in range(150):  # ~3.5 seconds of pure decode
            with torch.no_grad():
                model(next_token, past_key_values=past_key_values, use_cache=True)
        torch.cuda.synchronize()         
    
    time.sleep(1.5)  # let NVML window forget the prefill
    decode_compute, decode_memory = sample_utilization_during(run_decode_loop)
    decode_mem_gb = torch.cuda.memory_allocated() / 1e9 

    results.append({
        "phase": "decode",
        "prompt_length": actual_length,
        "time_sec": decode_elapsed,
        "tokens_per_sec": 1 / decode_elapsed,
        "mem_allocated_gb": decode_mem_gb,
        "gpu_memory_util_pct": decode_memory
    })
    print(f"decode  | kv_length={actual_length} | time_taken_sec={decode_elapsed:.4f}s | tokens_per_sec={1 / decode_elapsed:.1f} tok/s | gpu_memory={decode_memory:.1f}% gpu memory util")

# ── Save results ──────────────────────────────────────────────────────────────
# Save as CSV so the analysis notebook can load it without re-running.
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
df.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved to {RESULTS_PATH}")
