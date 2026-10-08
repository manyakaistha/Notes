"""Generate all Week 1 report figures with seaborn styling."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

OUT = Path("month 1/week 1/assets")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. Sampling rate
duration = 1.0
true_time = np.linspace(0, duration, 1000)
true_signal = np.sin(2 * np.pi * 2 * true_time)

sr_low = 10
t_low = np.linspace(0, duration, sr_low)
s_low = np.sin(2 * np.pi * 2 * t_low)

sr_high = 50
t_high = np.linspace(0, duration, sr_high)
s_high = np.sin(2 * np.pi * 2 * t_high)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(true_time, true_signal, color="grey", alpha=0.9, lw=2, label="True biology (continuous 2 Hz wave)")
axes[0].scatter(t_low, s_low, s=90, color="#d62728", zorder=5, label=f"Sampled @ {sr_low} Hz (10 dots)")
axes[0].plot(t_low, s_low, color="#d62728", alpha=0.4, ls="--")
axes[0].set_title(f"Low sampling rate ({sr_low} Hz): jagged, lossy", fontsize=15)
axes[0].set_ylabel("Voltage (a.u.)")
axes[0].legend(fontsize=11, loc="upper right")

axes[1].plot(true_time, true_signal, color="grey", alpha=0.9, lw=2, label="True biology (continuous 2 Hz wave)")
axes[1].scatter(t_high, s_high, s=45, color="#1f77b4", zorder=5, label=f"Sampled @ {sr_high} Hz (50 dots)")
axes[1].plot(t_high, s_high, color="#1f77b4", alpha=0.35, ls="--")
axes[1].set_title(f"High sampling rate ({sr_high} Hz): faithful trace", fontsize=15)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Voltage (a.u.)")
axes[1].legend(fontsize=11, loc="upper right")
fig.suptitle("Sampling rate changes the DATA, not the biology", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "sampling_rate_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. Synthetic single-channel EEG
sr, dur = 250, 2.0
n = int(sr * dur)
time = np.linspace(0, dur, n, endpoint=False)
delta = 15.0 * np.sin(2 * np.pi * 2 * time)
alpha = 8.0 * np.sin(2 * np.pi * 10 * time)
beta = 3.0 * np.sin(2 * np.pi * 22 * time)
noise = rng.normal(0, 1.5, n)
single = delta + alpha + beta + noise

fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for ax, comp, title, c in zip(
    axes,
    [delta, alpha, beta, single],
    ["Delta component — 2 Hz, 15 µV", "Alpha component — 10 Hz, 8 µV",
     "Beta component — 22 Hz, 3 µV", "Combined synthetic EEG (Delta + Alpha + Beta + noise)"],
    ["#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4"],
):
    sns.lineplot(x=time, y=comp, ax=ax, color=c, lw=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("µV")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Synthetic EEG: slow surge carries faster rhythms + noise", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "synthetic_eeg_trace.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. Five frequency bands
bands = [("Delta", 2, 15.0, "Deep non-REM sleep", "#2ca02c"),
         ("Theta", 6, 8.0, "Drowsiness / memory", "#17becf"),
         ("Alpha", 10, 8.0, "Eyes-closed rest", "#ff7f0e"),
         ("Beta", 22, 3.0, "Focus / motor", "#d62728"),
         ("Gamma", 45, 1.5, "Binding / higher-order", "#9467bd")]
tb = np.linspace(0, 1.0, 500, endpoint=False)
fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
for ax, (name, f, amp, ctx, c) in zip(axes, bands):
    y = amp * np.sin(2 * np.pi * f * tb)
    sns.lineplot(x=tb, y=y, ax=ax, color=c, lw=1.4)
    ax.set_title(f"{name}: ~{f} Hz — {ctx}", fontsize=13, loc="left")
    ax.set_ylabel("µV")
    ax.set_ylim(-17, 17)
axes[-1].set_xlabel("Time (s, 1-second window)")
fig.suptitle("The five classic EEG frequency bands (same 1 s window, same amplitude scale)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "frequency_bands.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. Multichannel EEG
n_ch = 8
multi = np.zeros((n_ch, n))
for ch in range(n_ch):
    wd = rng.uniform(0.5, 1.5)
    wa = rng.uniform(0.2, 1.8)
    ch_noise = rng.normal(0, 2.0, n)
    multi[ch, :] = wd * delta + wa * alpha + ch_noise

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(multi, cmap="coolwarm", center=0, cbar_kws={"label": "µV"},
            xticklabels=100, yticklabels=[f"Ch{i+1}" for i in range(n_ch)], ax=ax)
ax.set_title("Multichannel EEG matrix: 8 channels × 500 samples (250 Hz × 2 s)", fontsize=14)
ax.set_xlabel("Time sample index (columns = time)")
ax.set_ylabel("Channel (rows = space)")
fig.tight_layout()
fig.savefig(OUT / "multichannel_eeg_heatmap.png")
plt.close(fig)

fig, axes = plt.subplots(n_ch, 1, figsize=(11, 9), sharex=True)
t_plot = time[:250]
for i, ax in enumerate(axes):
    sns.lineplot(x=t_plot, y=multi[i, :250], ax=ax, color="#1f77b4", lw=1.1)
    ax.set_ylabel(f"Ch{i+1}", rotation=0, labelpad=20)
    ax.tick_params(labelleft=False)
axes[-1].set_xlabel("Time (s, first second only)")
fig.suptitle("Stacked traces: same rhythms, different per-channel weightings", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "multichannel_eeg_traces.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Synchrony vs amplitude
t = np.linspace(0, 0.5, 500)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
# synchronous: 5 neurons in phase
sync = np.sum([np.sin(2 * np.pi * 10 * t + 0) for _ in range(5)], axis=0)
for k in range(5):
    axes[0].plot(t, np.sin(2 * np.pi * 10 * t) + k * 0.15, color="#2ca02c", alpha=0.35, lw=1)
axes[0].plot(t, sync, color="#2ca02c", lw=2.5, label="Summed scalp signal (large)")
axes[0].set_title("Synchronous firing\n→ large amplitude", fontsize=13)
axes[0].set_xlabel("Time (s)")
axes[0].legend(fontsize=10)
# asynchronous: random phases cancel
phases = rng.uniform(0, 2 * np.pi, 5)
async_sum = np.sum([np.sin(2 * np.pi * 10 * t + p) for p in phases], axis=0)
for p in phases:
    axes[1].plot(t, np.sin(2 * np.pi * 10 * t + p), alpha=0.35, lw=1, color="#d62728")
axes[1].plot(t, async_sum, color="#d62728", lw=2.5, label="Summed scalp signal (small)")
axes[1].set_title("Asynchronous firing\n→ low amplitude (cancellation)", fontsize=13)
axes[1].set_xlabel("Time (s)")
axes[1].legend(fontsize=10)
fig.suptitle("Amplitude = synchrony, NOT raw computation", fontsize=15, y=1.05)
fig.tight_layout()
fig.savefig(OUT / "amplitude_synchrony.png")
plt.close(fig)

# ---------------------------------------------------------------- 6. Modality resolution trade-off
modalities = ["EEG", "ECoG", "MEG", "MRI", "fMRI", "Spikes"]
spatial = [1, 3, 2, 5, 5, 6]   # 1=poor ... 6=exact
temporal = [6, 6, 6, 0.5, 2, 6]  # 0.5=static/none
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
fig, ax = plt.subplots(figsize=(9, 6))
for m, s, t_, c in zip(modalities, spatial, temporal, colors):
    ax.scatter(s, t_, s=320, color=c, alpha=0.85, edgecolors="black")
    ax.text(s + 0.08, t_ + 0.08, m, fontsize=13, weight="bold")
ax.set_xlim(0.3, 6.8)
ax.set_ylim(0, 6.8)
ax.set_xlabel("Spatial resolution (poor → exact single neuron)")
ax.set_ylabel("Temporal resolution (static/slow → millisecond)")
ax.set_title("Neural modalities: no free lunch (space vs. time)", fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "modality_resolution_tradeoff.png")
plt.close(fig)

# ---------------------------------------------------------------- 7. EEG matrix shape schematic
demo = rng.normal(0, 1, size=(8, 60))
demo[2, 10:25] += 2.5  # fake burst
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(demo, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "µV"},
            xticklabels=10, yticklabels=[f"Ch{i+1}" for i in range(8)])
ax.set_title("EEG as AI sees it: a [Channels × Time] numbers matrix (here 8 × 60)", fontsize=14)
ax.set_xlabel("Time → (columns, 250 samples = 1 s @ 250 Hz)")
ax.set_ylabel("Space → (rows, one per electrode)")
fig.tight_layout()
fig.savefig(OUT / "eeg_matrix_shape.png")
plt.close(fig)

# ---------------------------------------------------------------- 8. Data-volume illustration
configs = ["32 ch\n250 Hz\n10 s", "64 ch\n500 Hz\n60 s"]
totals = [32 * 2500, 64 * 30000]
fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(x=configs, y=totals, hue=configs, palette=["#17becf", "#1f77b4"], legend=False, ax=ax)
ax.set_ylabel("Total samples (data points)")
ax.set_title("Same [C × T] idea — very different data volumes", fontsize=14)
for i, v in enumerate(totals):
    ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "eeg_data_volume.png")
plt.close(fig)

print("Saved figures:")
for p in sorted(OUT.glob("*.png")):
    print(" -", p, f"({p.stat().st_size//1024} KB)")
