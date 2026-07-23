# llm-inference-profiling

Profiling inference on Llama-3.1-8B-Instruct (fp16, single GPU) to understand where the time and memory actually go: prefill versus decode, which operations dominate each phase, how the KV cache grows, and what batching buys you before the GPU runs out of room. Four experiments, in order, build on each other. Results, plots, and raw traces are checked into this repo so the numbers below can be reproduced or checked directly.

## Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct`, loaded in fp16 through `transformers` with `device_map="auto"`.
- You need a Hugging Face token with access to the gated Llama checkpoint. Put it in a `.env` file as `hf_token=...`; `setup/model_download.py` reads it and logs in before pulling the model.
- GPU memory: the OOM boundaries in Experiment 3 land right around 22 to 24 GB of allocated memory, so these runs were done on a GPU in that class (not recorded explicitly anywhere, inferred from where things break).
- Dependencies are pinned in `setup/requirements.txt`: PyTorch 2.12, Transformers 5.8, `nvidia-ml-py` for GPU telemetry, `torch-model-profiler`, matplotlib, pandas.

Install and run:

```bash
pip install -r setup/requirements.txt
python experiments/01_prefill_vs_decode.py
python experiments/02_per_layer_profile.py
python experiments/03_kv_cache_memory.py
python experiments/04_batch_throughput.py
python analysis/04_analysis.py   # builds the plots for experiment 4
```

Each experiment script writes its own CSVs to `results/`. The profiler script also writes Chrome trace JSON files to `results/traces/`, which open directly in `chrome://tracing` or at ui.perfetto.dev.

## Repository layout

```
experiments/
  01_prefill_vs_decode.py     prefill vs decode timing, GPU utilization, memory
  02_per_layer_profile.py     PyTorch Profiler breakdown, per-operation CUDA time
  03_kv_cache_memory.py       KV cache growth, theory vs measured, OOM boundaries
  04_batch_throughput.py      batch size sweep: latency, throughput, memory
analysis/
  04_analysis.py              builds the four plots for experiment 4
results/                      CSVs and a PNG per experiment, plus raw traces
plots/                        PNG plots, and a scratch notebook for experiment 1
setup/
  model_download.py           loads model + tokenizer from the gated HF checkpoint
  requirements.txt
```

## Experiment 1: prefill vs decode

Goal: see how prefill and decode scale differently as prompt length grows, and check GPU utilization during each phase.

Method: for prompt lengths of 128, 512, 1024, 2048, and 4096 tokens, prefill ran five times and decode ran ten (medians reported, using the KV cache the prefill step produced). Separately, GPU utilization was sampled at 1ms intervals in a background thread via NVML, once during prefill and once over a longer 150-step decode loop (the extra length gives the sampler a real window to work with). A 1.5-second pause sits between the two passes so NVML's internal averaging window forgets the prefill burst before decode starts.

Results:

| Prompt length | Prefill tok/s | Decode tok/s | Prefill mem-util % | Decode mem-util % |
|---:|---:|---:|---:|---:|
| 128  | 3,114 | 44.0 | 59 | 94 |
| 512  | 3,867 | 43.7 | 33 | 93 |
| 1024 | 3,877 | 43.3 | 26 | 94 |
| 2048 | 3,851 | 42.1 | 26 | 94 |
| 4096 | 3,998 | 40.3 | 30 | 95 |

(Mem-util % here is NVML's memory-controller reading, not compute occupancy. The sampler tracks both, but only the memory figure ends up in the CSV.)

Two things stand out. Prefill throughput barely moves once the prompt passes a couple hundred tokens. The model is processing many tokens as one large matrix multiply, so it is compute-bound and already saturating the GPU at fairly modest lengths. Decode tells a different story: its tokens-per-second rate stays close to constant no matter how long the prompt was, yet memory-controller utilization sits at 93 to 95 percent throughout, well above anything prefill shows. That is the signature of a memory-bandwidth-bound step. Generating one token means reading the full set of weights plus the entire KV cache, and there is not much arithmetic to hide that read behind. The mild slide in decode speed, 44 down to 40 tok/s as the cache grows from 128 to 4096 tokens, fits the same story: more cache to read each step, slightly slower per step.

One rough edge worth flagging honestly: the length-128 prefill mem-util reading (59%) breaks the otherwise falling pattern across 512 to 2048. Short runs are more exposed to kernel launch overhead and sampling noise, so that single point should not be read as a real trend.

## Experiment 2: per-operation profiling

Goal: break one prefill step and one decode step down into per-operation CUDA time using `torch.profiler`, to see which kernels actually produce the difference above.

Method: five decode steps and three prefill steps (fixed at a 512-token prompt) ran inside `torch.profiler.profile`, with `record_function` markers separating the two phases. Output lands in `results/02_decode_profile_table.csv`, `results/02_prefill_profile_table.csv`, and Chrome traces under `results/traces/`.

The operation tables back up what the utilization numbers implied. Decode time is dominated by GEMV kernels, matrix-vector products, since a single new token per step collapses the attention and projection matmuls down to vector-matrix operations. Prefill time instead runs through GEMM kernels from cutlass's tensor-op path, the batched matrix-matrix kernels Tensor Cores are built for. Same weights, same model, but two different kernel families get invoked, purely because the token dimension of the input changes how the GPU schedules the work.

## Experiment 3: KV cache memory

Goal: check a hand-derived KV cache memory formula against what actually gets allocated, then find the batch size and sequence length where a fixed GPU runs out of room.

Formula, for Llama-3-8B's grouped-query attention layout (32 layers, 8 KV heads, not 32, head dim 128, fp16):

```
KV cache bytes = 2 (K and V) x num_kv_heads x head_dim x num_layers x seq_len x batch_size x 2 (fp16 bytes)
```

Theory vs. measured, batch size 1:

| Sequence length | Predicted (GB) | Measured (GB) | Error |
|---:|---:|---:|---:|
| 512  | 0.0671 | 0.0671 | 0.03% |
| 1024 | 0.1342 | 0.1342 | 0.03% |
| 2048 | 0.2684 | 0.2685 | 0.03% |
| 4096 | 0.5369 | 0.5369 | 0.01% |
| 8192 | 1.0737 | 1.0739 | 0.02% |

The formula tracks measured memory to within a few hundredths of a percent across a sixteenfold range of sequence lengths. Model weights alone come to roughly 16.1 GB allocated right after load, close to the back-of-envelope figure for an 8-billion-parameter model at 2 bytes per parameter (about 16 GB). That baseline barely moves; the KV cache is what grows on top of it.

OOM boundary, fixed prompt length of 2048 tokens, sweeping batch size:

| Batch size | Total memory (GB) | KV cache (GB) | Status |
|---:|---:|---:|---:|
| 1  | 16.9 | 0.79 | ok |
| 2  | 17.7 | 1.59 | ok |
| 4  | 19.3 | 3.18 | ok |
| 8  | 22.4 | 6.35 | ok |
| 16 | - | - | OOM |

And sweeping sequence length at batch size 1:

| Sequence length | Total memory (GB) | Status |
|---:|---:|---:|
| 16,384 | 22.4 | ok |
| 32,768 | - | OOM |
| 65,536 | - | OOM |

Both sweeps hit the same wall, somewhere around 22 to 24 GB allocated, whether that memory goes toward more concurrent requests or a longer context on a single one. The KV cache runs out first, not the model weights, and it grows linearly in both directions. Worth pointing out: using 8 KV heads instead of the model's 32 attention heads (grouped-query attention) is doing real work here. Standard multi-head attention would make this cache four times larger, and both OOM walls above would arrive roughly four times sooner.

## Experiment 4: batch throughput

Goal: at a fixed 512-token prompt, sweep batch size from 1 to 32 and measure decode-step latency (p50, p99) plus resulting throughput, to see what batching costs one user against what it earns in aggregate.

| Batch size | p50 (ms) | p99 (ms) | Per-request tok/s | Total tok/s | Memory (GB) |
|---:|---:|---:|---:|---:|---:|
| 1  | 35.2 | 243.6 | 28.4 | 28.4  | 16.2 |
| 2  | 35.7 | 53.4  | 28.0 | 56.0  | 16.2 |
| 4  | 35.9 | 40.1  | 27.9 | 111.6 | 16.4 |
| 8  | 35.6 | 38.3  | 28.1 | 224.7 | 16.7 |
| 16 | 35.3 | 43.8  | 28.3 | 452.8 | 17.3 |
| 32 | 41.5 | 62.0  | 24.1 | 771.8 | 18.5 |

From batch 1 through 16, p50 latency holds flat around 35 ms no matter how many requests share the GPU, and total throughput scales almost linearly with batch size (16x the batch gives close to 16x the tokens per second). That is the batching argument in one table: running 16 requests together here costs barely more per request than running one alone. Batch 1's p99 of 243 ms looks like an outlier rather than a real latency ceiling, probably a cold-cache or scheduling artifact on the first measured batch, since every larger batch drops p99 straight back down to the 40 to 60 ms range.

Batch 32 is where the free scaling ends. Per-request throughput drops from about 28 to 24 tok/s, and p50 rises to 41.5 ms. The GPU's compute is no longer sitting idle waiting on memory bandwidth, so more concurrent sequences start to cost each one something. Total throughput keeps climbing (772 tok/s) but not at the earlier linear rate. Memory grows the whole time too, from 16.2 GB at batch 1 to 18.5 GB at batch 32, matching the linear KV-cache-per-request growth measured in Experiment 3.

Plots for all four relationships (latency vs. batch, throughput vs. batch, per-request vs. total throughput, memory vs. batch) live in `plots/`, built by `analysis/04_analysis.py`.

## Putting it together

The four experiments describe one picture, not four separate ones. Prefill is compute-bound and decode is memory-bandwidth-bound; Experiments 1 and 2 agree on this from two different angles, utilization numbers and kernel-level profiling. The KV cache formula from Experiment 3 predicts memory almost exactly, and that in turn explains why Experiment 4's batching curve looks the way it does. Latency stays flat because decode was memory-bound to begin with, so extra requests share the same memory traffic instead of competing for it, at least until compute becomes the bottleneck around batch 32. And the OOM walls from Experiment 3 set the ceiling on how far that batching curve can go on a single GPU of this size.