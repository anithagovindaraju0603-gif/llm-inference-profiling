import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("results/04_batch_throughput.csv")
df_ok = df[df["status"] == "ok"].copy()

os.makedirs("plots", exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_GRAY   = "#999999"

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Latency vs Batch Size (p50 and p99)
# Story: p50 is nearly flat → latency barely grows with more users
#        p99 shows occasional spikes → variance, not average cost
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_ok["batch_size"], df_ok["p50_ms"],
        marker="o", linewidth=2.5, color=C_BLUE, label="p50 (median latency)")
ax.plot(df_ok["batch_size"], df_ok["p99_ms"],
        marker="s", linewidth=2.5, linestyle="--", color=C_ORANGE, label="p99 (worst-case latency)")

# Annotate each p50 point
for _, row in df_ok.iterrows():
    ax.annotate(f'{row["p50_ms"]:.1f}ms',
                xy=(row["batch_size"], row["p50_ms"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color=C_BLUE)

ax.set_xlabel("Batch Size (number of concurrent users)", fontsize=11)
ax.set_ylabel("Decode Step Latency (ms)", fontsize=11)
ax.set_title("Per-User Latency vs Batch Size\nLatency stays flat — each user doesn't wait longer with more users", fontsize=12)
ax.set_xticks(df_ok["batch_size"])
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, df_ok["p99_ms"].max() * 1.2)

plt.tight_layout()
plt.savefig("plots/04_latency_vs_batch.png", dpi=150)
plt.show()
print("Saved → plots/04_latency_vs_batch.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Total Throughput vs Batch Size (actual vs ideal linear)
# Story: bars nearly touch the dotted ideal line → near-perfect linear scaling
#        batch 32 falls short → GPU compute starting to saturate
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# Ideal linear scaling anchored at batch 1
baseline_tps = df_ok.loc[df_ok["batch_size"] == 1, "total_throughput_tps"].values[0]
ideal = df_ok["batch_size"] * baseline_tps

ax.bar(df_ok["batch_size"], df_ok["total_throughput_tps"],
       color=C_BLUE, alpha=0.85, label="Actual throughput", width=2.5)
ax.plot(df_ok["batch_size"], ideal,
        linestyle="--", color=C_ORANGE, linewidth=2, label="Ideal linear scaling")

# Annotate each bar
for _, row in df_ok.iterrows():
    ax.text(row["batch_size"], row["total_throughput_tps"] + 8,
            f'{row["total_throughput_tps"]:.0f}',
            ha="center", fontsize=8, color="white" if row["batch_size"] < 32 else C_BLUE)

ax.set_xlabel("Batch Size (number of concurrent users)", fontsize=11)
ax.set_ylabel("Total Tokens per Second (all users combined)", fontsize=11)
ax.set_title("Total Throughput vs Batch Size\nNearly linear scaling — batching is almost free throughput", fontsize=12)
ax.set_xticks(df_ok["batch_size"])
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/04_throughput_vs_batch.png", dpi=150)
plt.show()
print("Saved → plots/04_throughput_vs_batch.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Per-request TPS vs Total Throughput (the money chart)
# Story: one line flat, one climbing → same cost per user, more output overall
#        this is WHY production serving engines obsess over batching
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_ok["batch_size"], df_ok["tps_per_request"],
        marker="o", linewidth=2.5, color=C_BLUE, label="Per-user throughput (tok/s per request)")
ax.plot(df_ok["batch_size"], df_ok["total_throughput_tps"],
        marker="s", linewidth=2.5, color=C_GREEN, label="Total throughput (tok/s across all users)")

# Shade the gap between them — the "free gains" region
ax.fill_between(df_ok["batch_size"],
                df_ok["tps_per_request"],
                df_ok["total_throughput_tps"],
                alpha=0.08, color=C_GREEN, label="Efficiency gain from batching")

ax.set_xlabel("Batch Size (number of concurrent users)", fontsize=11)
ax.set_ylabel("Tokens per Second", fontsize=11)
ax.set_title("Per-User vs Total Throughput\nFlat line + rising line = why batching is everything in production", fontsize=12)
ax.set_xticks(df_ok["batch_size"])
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/04_per_request_vs_total.png", dpi=150)
plt.show()
print("Saved → plots/04_per_request_vs_total.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Memory growth vs batch size
# Story: memory grows linearly with batch size (each user needs their own KV cache)
#        ties back to exp 3 — this is the wall that stops you from batching forever
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_ok["batch_size"], df_ok["mem_allocated_gb"],
        marker="o", linewidth=2.5, color=C_ORANGE)
ax.fill_between(df_ok["batch_size"], df_ok["mem_allocated_gb"],
                alpha=0.15, color=C_ORANGE)

# Draw the VRAM ceiling
VRAM_GB = 24
ax.axhline(y=VRAM_GB, color="red", linestyle="--", linewidth=1.5, label=f"GPU VRAM ceiling ({VRAM_GB}GB)")
ax.text(df_ok["batch_size"].max(), VRAM_GB + 0.3, "VRAM limit", color="red", fontsize=9, ha="right")

# Annotate each point
for _, row in df_ok.iterrows():
    ax.annotate(f'{row["mem_allocated_gb"]:.2f}GB',
                xy=(row["batch_size"], row["mem_allocated_gb"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color=C_ORANGE)

ax.set_xlabel("Batch Size (number of concurrent users)", fontsize=11)
ax.set_ylabel("GPU Memory Allocated (GB)", fontsize=11)
ax.set_title("Memory Growth vs Batch Size\nLinear growth — KV cache per user adds up until you hit the wall", fontsize=12)
ax.set_xticks(df_ok["batch_size"])
ax.set_ylim(0, VRAM_GB * 1.15)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/04_memory_vs_batch.png", dpi=150)
plt.show()
print("Saved → plots/04_memory_vs_batch.png")

print("\nAll 4 plots saved to plots/")