"""Deterministic Week 2 Transformer figures; CPU-only NumPy toy examples."""
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
OUT = Path(__file__).parent / "week 2" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(f"[Generated] {OUT / name}")


def pe(pos, d):
    pos = np.asarray(pos)[:, None]
    i = np.arange(d // 2)
    phase = pos / (10000 ** (2 * i / d))[None, :]
    out = np.empty((len(pos), d))
    out[:, 0::2], out[:, 1::2] = np.sin(phase), np.cos(phase)
    return out


def main():
    # 01: sinusoidal positional encoding heatmap.
    enc = pe(np.arange(100), 64)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(enc.T, cmap="vlag", center=0, xticklabels=10, yticklabels=8, cbar_kws={"label": "PE value (unitless)"}, ax=ax)
    ax.set(xlabel="Token position (index)", ylabel="Embedding dimension (index)", title="Sinusoidal positional encoding [position, dimension]")
    save(fig, "01_sinusoidal_pe_heatmap.png")

    # 02: every sine/cosine pair rotates by offset k; verify one block matrix.
    d, pos, shift = 64, np.arange(120), 7
    P, Pshift = pe(pos, d), pe(pos + shift, d)
    w = 10000 ** (-2 * np.arange(d // 2) / d)
    mats = np.zeros((d, d))
    for j, angle in enumerate(w * shift):
        c, s = np.cos(angle), np.sin(angle)
        mats[2*j:2*j+2, 2*j:2*j+2] = [[c, s], [-s, c]]
    err = np.max(np.abs(P @ mats.T - Pshift))
    offsets = np.arange(0, 41)
    dot = np.array([np.mean(np.sum(pe([20], d) * pe([20+k], d), axis=1)) for k in offsets])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(pos, np.linalg.norm(P @ mats.T - Pshift, axis=1), color="C0")
    axes[0].set(xlabel="Base position (index)", ylabel="Rotation identity error (unitless)", title=rf"$PE(pos+{shift})=M_{shift}PE(pos)$; max error={err:.1e}")
    axes[1].plot(offsets, dot, "o-")
    axes[1].set(xlabel="Relative offset k (tokens)", ylabel="Mean PE dot product (unitless)", title="Dot product depends on relative offset")
    fig.tight_layout()
    save(fig, "02_pe_relative_offset.png")

    # 03: normalization axes schematic for [B,T,d].
    x = np.arange(2*3*4, dtype=float).reshape(2,3,4) + rng.normal(scale=.4, size=(2,3,4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, xlabel, ylabel in [(axes[0], x[0], "LayerNorm: each token, across d", "Feature d (normalized axis)", "Token T"), (axes[1], x.reshape(6,4), "BatchNorm: each feature, across B × T", "Feature d", "Batch × token position (normalized axis)")]:
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="vlag", center=mat.mean(), ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.suptitle("Normalization axes on an activation tensor [B,T,d] = [2,3,4]")
    fig.tight_layout()
    save(fig, "03_norm_axes.png")

    # 04: Vaswani inverse-square-root schedule.
    steps = np.arange(1, 16001)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for dm in (512, 256):
        lr = dm**-0.5 * np.minimum(steps**-0.5, steps * 4000**-1.5)
        ax.plot(steps, lr, label=fr"$d_{{model}}={dm}$")
    ax.axvline(4000, color="gray", linestyle="--", label="warmup = 4000 steps")
    ax.set(xlabel="Training step (updates)", ylabel="Learning rate (unitless)", title=r"Transformer schedule: $d_{model}^{-1/2}\min(step^{-1/2}, step\,warmup^{-3/2})$", xscale="log", yscale="log")
    ax.legend()
    fig.tight_layout()
    save(fig, "04_warmup_schedule.png")

    # 05: causal mask and resulting probability matrix.
    scores = np.array([[1.2, .5, 2.1, -.3], [.2, 1.0, .8, 1.1], [.4, -.2, 1.4, .7], [.6, .3, -.1, 1.3]])
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    masked = scores.copy(); masked[mask] = -np.inf
    e = np.exp(masked - np.max(masked, axis=1, keepdims=True)); probs = e / e.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.heatmap(np.where(mask, 1, 0), annot=True, fmt=".0f", cmap="Reds", cbar=False, ax=axes[0])
    axes[0].set(title="Future positions masked (1 = blocked)", xlabel="Key position (index)", ylabel="Query position (index)")
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, ax=axes[1])
    axes[1].set(title="Softmax after $-\infty$ masking", xlabel="Key position (index)", ylabel="Query position (index)")
    fig.tight_layout()
    save(fig, "05_causal_mask_softmax.png")

    # 06: smoothing targets and cross-entropy as confidence varies.
    eps, vocab, target_id = .1, 5, 2
    target = np.full(vocab, eps/vocab); target[target_id] += 1-eps
    confidence = np.linspace(.2, .99, 120)
    losses = []
    for c in confidence:
        p = np.full(vocab, (1-c)/(vocab-1)); p[target_id] = c
        losses.append(-np.sum(target * np.log(p)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(np.arange(vocab), target, color=["C0" if i != target_id else "C1" for i in range(vocab)])
    axes[0].set(xticks=np.arange(vocab), xlabel="Vocabulary class (index)", ylabel="Target probability (unitless)", title=r"Smoothed target, $\epsilon=0.1$, $V=5$")
    axes[1].plot(confidence, losses)
    axes[1].set(xlabel="Model probability on correct class (unitless)", ylabel="Cross-entropy loss (nats)", title="Smoothed cross-entropy vs confidence")
    fig.tight_layout()
    save(fig, "06_label_smoothing.png")


if __name__ == "__main__":
    main()
