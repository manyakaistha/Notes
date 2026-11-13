"""Deterministic Week 1 Transformer figures; CPU-only NumPy toy examples."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(42)
rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
OUT = Path(__file__).parent / "week 1" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(f"[Generated] {OUT / name}")


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    # 01: toy sentence attention, scaled dot product.
    tokens = ["The", "cat", "sat", "on", "mat"]
    x = rng.normal(size=(5, 16))
    q, k = x @ rng.normal(size=(16, 8)), x @ rng.normal(size=(16, 8))
    weights = softmax(q @ k.T / np.sqrt(8))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="mako", xticklabels=tokens, yticklabels=tokens, vmin=0, vmax=1, ax=ax)
    ax.set(xlabel="Key token position (index)", ylabel="Query token position (index)", title=r"Scaled dot-product attention $QK^T/\sqrt{d_k}$")
    save(fig, "01_scaled_attention_heatmap.png")

    # 02: random unit-variance coordinates give Var(q.k)=d_k; scaling fixes it.
    dims = np.array([4, 16, 64, 256])
    trials = 30000
    raw_std, scaled_std, p_raw, p_scaled, g_raw, g_scaled = [], [], [], [], [], []
    for d in dims:
        a, b = rng.normal(size=(trials, d)), rng.normal(size=(trials, d))
        dots = np.sum(a * b, axis=1)
        raw_std.append(dots.std())
        scaled_std.append((dots / np.sqrt(d)).std())
        logits = np.stack([dots, np.zeros_like(dots)], axis=1)
        logits_s = logits / np.sqrt(d)
        p_raw.append(softmax(logits).max(axis=1).mean())
        p_scaled.append(softmax(logits_s).max(axis=1).mean())
        pr, ps = softmax(logits)[:, 0], softmax(logits_s)[:, 0]
        g_raw.append(np.mean(pr * (1 - pr)))
        g_scaled.append(np.mean(ps * (1 - ps)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].plot(dims, raw_std, "o-", label="unscaled")
    axes[0].plot(dims, scaled_std, "s-", label=r"scaled by $1/\sqrt{d_k}$")
    axes[0].set(xlabel=r"Key dimension $d_k$ (components)", ylabel="Dot-product SD (unitless)", title=r"$q^Tk$ scale")
    axes[0].legend(fontsize=9)
    axes[1].plot(dims, p_raw, "o-", label="unscaled")
    axes[1].plot(dims, p_scaled, "s-", label="scaled")
    axes[1].set(xlabel=r"$d_k$ (components)", ylabel="Mean max probability (unitless)", title="Softmax confidence")
    axes[2].plot(dims, g_raw, "o-", label="unscaled")
    axes[2].plot(dims, g_scaled, "s-", label="scaled")
    axes[2].set(xlabel=r"$d_k$ (components)", ylabel=r"Mean $p(1-p)$ (unitless)", title="Logit derivative magnitude")
    for ax in axes[1:]: ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "02_scaling_saturation.png")

    # 03: contrast one fixed context with a learned additive alignment.
    enc = np.array([[1.0, 0.2], [0.1, 1.0], [0.7, 0.6], [-0.4, 0.8], [0.9, -0.2]])
    dec = np.array([[0.8, 0.1], [0.0, 0.9], [-0.3, 0.7], [0.6, -0.4]])
    W, U, v = rng.normal(size=(2, 4)), rng.normal(size=(2, 4)), rng.normal(size=4)
    score = np.tanh(dec[:, None, :] @ U + enc[None, :, :] @ W) @ v
    alpha = softmax(score)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.ones((4, 5)) / 5, cmap="mako", vmin=0, vmax=1, aspect="auto")
    axes[0].set(title="Fixed-vector bottleneck\nOne context reused at every output", xlabel="Encoder position (index)", ylabel="Decoder step (index)")
    sns.heatmap(alpha, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, ax=axes[1])
    axes[1].set(title=r"Bahdanau alignment $\alpha_{ti}$", xlabel="Encoder position (index)", ylabel="Decoder step (index)")
    fig.tight_layout()
    save(fig, "03_seq2seq_vs_alignment.png")

    # 04: multi-head attention patterns and tensor shape flow.
    seq = rng.normal(size=(6, 32))
    patt = []
    for h in range(4):
        qh = seq @ rng.normal(size=(32, 8))
        kh = seq @ rng.normal(size=(32, 8))
        patt.append(softmax(qh @ kh.T / np.sqrt(8)))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for i, ax in enumerate(axes[:4]):
        sns.heatmap(patt[i], cmap="mako", vmin=0, vmax=1, cbar=False, ax=ax)
        ax.set(title=f"Head {i+1}", xlabel="Key position (index)", ylabel="Query position (index)" if i == 0 else "")
    axes[4].barh([3, 2, 1, 0], [512, 512, 512, 512], color=sns.color_palette("deep", 4))
    axes[4].set_yticks([3,2,1,0], ["Input", "Q/K/V", "Heads", "Concat + Wᴼ"])
    axes[4].set(xlabel="Feature width (components)", title="Shape flow: [B,T,512]\n→ [B,8,T,64] → [B,T,512]", xlim=(0, 560))
    fig.tight_layout()
    save(fig, "04_multihead_patterns_shapes.png")

    # 05: parameter breakdown for base encoder; biases and two LayerNorms explicit.
    dm, df = 512, 2048
    components = {"MHA weights": 4*dm*dm, "MHA biases": 4*dm, "FFN weights": 2*dm*df, "FFN biases": df+dm, "2 LayerNorms": 4*dm}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(components.keys(), np.array(list(components.values())) / 1e6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(ylabel="Trainable parameters (millions)", xlabel="Encoder-layer component", title=f"Encoder layer parameters (d_model={dm}, d_ff={df})", ylim=(0, 2.4))
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    print(f"[Encoder] formula components={sum(components.values()):,}; MHA={components['MHA weights']+components['MHA biases']:,}; FFN={components['FFN weights']+components['FFN biases']:,}; LayerNorm={components['2 LayerNorms']:,}")
    save(fig, "05_encoder_parameter_breakdown.png")


if __name__ == "__main__":
    main()
