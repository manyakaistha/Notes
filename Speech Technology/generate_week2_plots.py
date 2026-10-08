"""
Deterministic figure generation for Week 2 of Speech Technology.

Style: Seaborn whitegrid, talk context. NumPy and Matplotlib only.
Outputs PNGs to 'week 2/assets/'. Each function owns its own RNG seed.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

np.random.seed(42)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "sans-serif"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "week 2", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(ASSETS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Generated] {path}")


# -----------------------------------------------------------------------------
# Figure 1: The Complete MFCC Extraction Pipeline
# -----------------------------------------------------------------------------
def plot_mfcc_pipeline():
    fig, axes = plt.subplots(4, 1, figsize=(13, 13))

    fs = 16000
    n_samples = int(0.025 * fs)  # 400 samples (25 ms)
    t = np.linspace(0, 25, n_samples)
    
    # Synthetic voiced frame with formant resonances
    f0 = 130.0
    signal = (
        np.sin(2 * np.pi * f0 * t / 1000)
        + 0.8 * np.sin(2 * np.pi * 3 * f0 * t / 1000)
        + 0.5 * np.sin(2 * np.pi * 6 * f0 * t / 1000)
        + 0.3 * np.sin(2 * np.pi * 18 * f0 * t / 1000)
    )
    window = np.hamming(n_samples)
    windowed = signal * window

    # 1. Windowed Time Frame
    axes[0].plot(t, signal, color="gray", linestyle="--", alpha=0.6, label="Raw speech frame")
    axes[0].plot(t, windowed, color="teal", lw=2, label="Hamming-windowed frame (25 ms)")
    axes[0].set_title("(a) Short-Time Speech Frame Windowing", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time (milliseconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=9)

    # 2. Short-Time Power Spectrum
    n_fft = 512
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    spec = np.abs(np.fft.rfft(windowed, n_fft)) ** 2
    spec_db = 10 * np.log10(spec + 1e-12)
    axes[1].plot(freqs, spec_db, color="darkblue", lw=1.8)
    axes[1].set_title(r"(b) Periodogram Power Spectrum $|X[k]|^2$ (DFT/FFT)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].set_xlim(0, 8000)

    # 3. Triangular Mel Filter Bank Integration
    n_mels = 26
    def hz_to_mel(f): return 1125.0 * np.log(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (np.exp(m / 1125.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(fs / 2), n_mels + 2)
    bin_pts = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / fs).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bin_pts[m - 1], bin_pts[m]):
            fbank[m - 1, k] = (k - bin_pts[m - 1]) / max(1, bin_pts[m] - bin_pts[m - 1])
        for k in range(bin_pts[m], bin_pts[m + 1]):
            fbank[m - 1, k] = (bin_pts[m + 1] - k) / max(1, bin_pts[m + 1] - bin_pts[m])

    filter_energies = np.dot(fbank, spec)
    log_energies = np.log(filter_energies + 1e-6)

    m_indices = np.arange(1, n_mels + 1)
    markerline, stemlines, _ = axes[2].stem(m_indices, log_energies, linefmt="darkgreen", markerfmt="o", basefmt="k-")
    plt.setp(markerline, color="darkgreen", markersize=6)
    plt.setp(stemlines, color="darkgreen", linewidth=1.5)
    axes[2].set_title(r"(c) Log Mel Filter Bank Energies $\ln(E_m)$ (26 Channels)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Mel Filter Channel Index ($m$)")
    axes[2].set_ylabel("Log Energy")
    axes[2].set_xlim(0, 27)

    # 4. DCT-II Coefficients
    num_ceps = 26
    mfcc_all = np.zeros(num_ceps)
    for n in range(num_ceps):
        mfcc_all[n] = np.sum(log_energies * np.cos(np.pi * n * (np.arange(n_mels) + 0.5) / n_mels))

    ceps_indices = np.arange(num_ceps)
    colors = ["crimson" if i < 13 else "gray" for i in ceps_indices]
    for i in range(num_ceps):
        axes[3].vlines(ceps_indices[i], 0, mfcc_all[i], color=colors[i], lw=2)
        axes[3].plot(ceps_indices[i], mfcc_all[i], "o", color=colors[i], ms=6)
    axes[3].axhline(0, color="black", lw=1)
    axes[3].axvline(12.5, color="black", linestyle="--", lw=1.5, alpha=0.7)
    axes[3].text(6, np.max(mfcc_all) * 0.8, "Retained MFCCs (c_0 to c_12)\nVocal tract envelope", 
                 ha="center", fontsize=10, fontweight="bold", color="crimson",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", lw=1))
    axes[3].text(19, np.max(mfcc_all) * 0.8, "Truncated Coefficients (c_13+)\nPitch & excitation detail", 
                 ha="center", fontsize=10, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
    axes[3].set_title(r"(d) Discrete Cosine Transform (DCT-2) Coefficients $c_n$", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Cepstral Coefficient Index ($n$)")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlim(-1, 26)

    fig.tight_layout()
    _save(fig, "01_mfcc_extraction_pipeline.png")


# -----------------------------------------------------------------------------
# Figure 2: Mel Scale Warping and Triangular Filter Bank Response
# -----------------------------------------------------------------------------
def plot_mel_warping():
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Panel 1: Mel Transformation Curve
    f_hz = np.linspace(0, 8000, 2000)
    f_mel = 1125.0 * np.log(1.0 + f_hz / 700.0)

    axes[0].plot(f_hz, f_mel, color="crimson", lw=2.5, label=r"$f_{\mathrm{mel}} = 1125 \ln(1 + f_{\mathrm{Hz}} / 700)$")
    axes[0].axvline(1000, color="gray", linestyle="--", lw=1.2, alpha=0.8)
    axes[0].plot([1000], [1125.0 * np.log(1.0 + 1000.0 / 700.0)], "ko", ms=7)
    axes[0].text(1050, 600, "Transition Boundary (~1000 Hz)\nLinear -> Logarithmic", fontsize=10, fontweight="bold")
    
    # Linear reference tangent at origin: slope = 1125 / 700 ~ 1.607
    axes[0].plot(f_hz[:500], (1125.0 / 700.0) * f_hz[:500], color="navy", linestyle=":", lw=1.8, label="Linear asymptote (slope 1.607)")
    axes[0].set_title("(a) Mel Scale Frequency Transformation vs. Physical Hertz", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Physical Frequency $f_{\mathrm{Hz}}$ (Hz)")
    axes[0].set_ylabel("Perceptual Frequency $f_{\mathrm{mel}}$ (Mels)")
    axes[0].set_xlim(0, 8000)
    axes[0].set_ylim(0, 3100)
    axes[0].legend(loc="lower right", fontsize=10)

    # Panel 2: Triangular Filter Bank
    n_filters = 26
    fs = 16000
    n_fft = 512
    def hz_to_mel(f): return 1125.0 * np.log(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (np.exp(m / 1125.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(fs / 2), n_filters + 2)
    hz_pts = mel_to_hz(mel_pts)
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)

    palette = sns.color_palette("mako", n_filters)
    for m in range(1, n_filters + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        resp = np.zeros_like(freqs)
        for i, f in enumerate(freqs):
            if left <= f <= center:
                resp[i] = (f - left) / (center - left)
            elif center <= f <= right:
                resp[i] = (right - f) / (right - center)
        axes[1].plot(freqs, resp, color=palette[m - 1], lw=1.6)

    axes[1].set_title("(b) 26-Channel Triangular Mel Filter Bank (Narrow at low f, Broad at high f)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Filter Gain")
    axes[1].set_xlim(0, 8000)
    axes[1].set_ylim(-0.05, 1.08)

    fig.tight_layout()
    _save(fig, "02_mel_warping_filterbank_geometry.png")


# -----------------------------------------------------------------------------
# Figure 3: DCT Feature Decorrelation (Covariance Matrix Comparison)
# -----------------------------------------------------------------------------
def plot_dct_decorrelation():
    np.random.seed(42)
    n_frames = 600
    n_mels = 20

    # Simulate realistic correlated Log Mel Filter Bank energies via AR(1) across frequency channels
    log_mel = np.zeros((n_frames, n_mels))
    for t in range(n_frames):
        # Base vocal tract envelope with 2 formants
        f1_bin = int(4 + 2 * np.sin(2 * np.pi * t / 100))
        f2_bin = int(12 + 3 * np.cos(2 * np.pi * t / 100))
        env = np.exp(-0.5 * (np.arange(n_mels) - f1_bin) ** 2 / 2.0) + 0.6 * np.exp(-0.5 * (np.arange(n_mels) - f2_bin) ** 2 / 3.0)
        # Add correlated noise across adjacent bins
        noise = np.random.randn(n_mels)
        corr_noise = np.convolve(noise, [0.3, 0.5, 0.3], mode="same")
        log_mel[t] = 5.0 * env + corr_noise

    corr_logmel = np.corrcoef(log_mel, rowvar=False)

    # Apply DCT-2 across frequency channels
    num_ceps = 13
    mfcc = np.zeros((n_frames, num_ceps))
    for n in range(num_ceps):
        mfcc[:, n] = np.sum(log_mel * np.cos(np.pi * n * (np.arange(n_mels) + 0.5) / n_mels), axis=1)

    corr_mfcc = np.corrcoef(mfcc, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # Heatmap 1: Log Mel Correlation
    sns.heatmap(corr_logmel, ax=axes[0], cmap="vlag", vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Correlation Coefficient", "shrink": 0.8})
    axes[0].set_title("(a) Log Mel Filter Bank Correlation Matrix\n(High off-diagonal cross-correlation)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Filter Bank Channel Index")
    axes[0].set_ylabel("Filter Bank Channel Index")

    # Heatmap 2: MFCC Correlation
    sns.heatmap(corr_mfcc, ax=axes[1], cmap="vlag", vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Correlation Coefficient", "shrink": 0.8})
    axes[1].set_title("(b) MFCC Correlation Matrix (Post-DCT)\n(Near-zero off-diagonals -> Diagonal Covariance)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Cepstral Coefficient Index")
    axes[1].set_ylabel("Cepstral Coefficient Index")

    fig.tight_layout()
    _save(fig, "03_dct_feature_decorrelation.png")


# -----------------------------------------------------------------------------
# Figure 4: Dynamic Features (Static, Delta, and Delta-Delta)
# -----------------------------------------------------------------------------
def plot_dynamic_deltas():
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    n_frames = 60
    t = np.arange(n_frames)
    
    # Simulate a smooth phonetic transition (e.g. vowel formant movement /a/ -> /i/)
    static_c1 = 10.0 + 8.0 / (1.0 + np.exp(-0.3 * (t - 28))) + 0.3 * np.sin(0.4 * t)

    # Compute delta with window K = 2
    K = 2
    delta_c1 = np.zeros(n_frames)
    denom = 2 * np.sum(np.arange(1, K + 1) ** 2)
    for n in range(K, n_frames - K):
        num = np.sum([k * (static_c1[n + k] - static_c1[n - k]) for k in range(1, K + 1)])
        delta_c1[n] = num / denom

    # Compute delta-delta from delta
    ddelta_c1 = np.zeros(n_frames)
    for n in range(2 * K, n_frames - 2 * K):
        num = np.sum([k * (delta_c1[n + k] - delta_c1[n - k]) for k in range(1, K + 1)])
        ddelta_c1[n] = num / denom

    # Panel 1: Static
    axes[0].plot(t, static_c1, color="navy", lw=2.2, label=r"Static $c_1(t)$")
    axes[0].axvline(28, color="gray", linestyle="--", lw=1.2, alpha=0.7)
    axes[0].text(28.5, 11, "Phonetic Transition Center", fontsize=9, fontweight="bold")
    axes[0].set_title(r"(a) Static Cepstral Trajectory $c_1(t)$", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Static Amplitude")
    axes[0].legend(loc="upper left", fontsize=9)

    # Panel 2: Delta
    axes[1].plot(t, delta_c1, color="darkgreen", lw=2.2, label=r"Delta $\Delta c_1(t)$ (Velocity)")
    axes[1].axvline(28, color="gray", linestyle="--", lw=1.2, alpha=0.7)
    axes[1].plot([28], [np.max(delta_c1)], "o", color="darkgreen", ms=7)
    axes[1].text(29, np.max(delta_c1) * 0.9, r"Max Velocity ($\Delta c_t$ peak)", fontsize=9, fontweight="bold")
    axes[1].set_title(r"(b) First Derivative: Velocity $\Delta c_1(t) = \frac{\sum k (c_{t+k} - c_{t-k})}{2 \sum k^2}$", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Velocity")
    axes[1].legend(loc="upper left", fontsize=9)

    # Panel 3: Delta-Delta
    axes[2].plot(t, ddelta_c1, color="crimson", lw=2.2, label=r"Delta-Delta $\Delta\Delta c_1(t)$ (Acceleration)")
    axes[2].axhline(0, color="black", lw=0.8, linestyle=":")
    axes[2].axvline(28, color="gray", linestyle="--", lw=1.2, alpha=0.7)
    axes[2].set_title(r"(c) Second Derivative: Acceleration $\Delta\Delta c_1(t)$ (Zero-Crossing at Inflection)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time Frame Index ($t$, 10 ms hop)")
    axes[2].set_ylabel("Acceleration")
    axes[2].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    _save(fig, "04_dynamic_spectral_deltas.png")


# -----------------------------------------------------------------------------
# Figure 5: K-Means Vector Quantization and LBG Centroid Splitting
# -----------------------------------------------------------------------------
def plot_kmeans_lbg():
    np.random.seed(10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Generate 4 synthetic vowel clusters in (F1, F2) space
    c1 = np.random.randn(80, 2) * [40, 90] + [300, 2200]   # /i/
    c2 = np.random.randn(80, 2) * [50, 70] + [500, 1800]   # /e/
    c3 = np.random.randn(80, 2) * [60, 60] + [750, 1200]   # /a/
    c4 = np.random.randn(80, 2) * [45, 50] + [350, 800]    # /u/
    X = np.vstack([c1, c2, c3, c4])

    global_mean = np.mean(X, axis=0)

    # Panel 1: Single Global Cluster
    axes[0].scatter(X[:, 0], X[:, 1], color="gray", alpha=0.45, s=25)
    axes[0].scatter([global_mean[0]], [global_mean[1]], color="crimson", s=180, marker="X", zorder=5, label=r"Global Centroid $\mu$")
    axes[0].set_title("(a) Codebook Size $K = 1$\n(Single Global Mean)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(r"Feature Dimension 1 ($F_1$ Hz)")
    axes[0].set_ylabel(r"Feature Dimension 2 ($F_2$ Hz)")
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: LBG Binary Splitting
    eps = 0.15
    split1 = global_mean * (1.0 + eps)
    split2 = global_mean * (1.0 - eps)
    axes[1].scatter(X[:, 0], X[:, 1], color="gray", alpha=0.45, s=25)
    axes[1].scatter([global_mean[0]], [global_mean[1]], color="black", s=100, marker="o", alpha=0.5, label="Prior Mean")
    axes[1].scatter([split1[0]], [split1[1]], color="teal", s=160, marker="^", zorder=5, label=r"$c_1 = \mu(1 + \epsilon)$")
    axes[1].scatter([split2[0]], [split2[1]], color="darkorange", s=160, marker="v", zorder=5, label=r"$c_2 = \mu(1 - \epsilon)$")
    axes[1].annotate("", xy=split1, xytext=global_mean, arrowprops=dict(arrowstyle="->", color="teal", lw=2))
    axes[1].annotate("", xy=split2, xytext=global_mean, arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
    axes[1].set_title(r"(b) LBG Binary Centroid Split\n$c_{1,2} = \mu(1 \pm \epsilon)$ ($\epsilon = 0.15$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(r"Feature Dimension 1 ($F_1$ Hz)")
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel 3: Converged K-Means (K=4)
    # Simple manual centroids matching true clusters
    means_final = np.array([[300, 2200], [500, 1800], [750, 1200], [350, 800]])
    labels = np.argmin(np.linalg.norm(X[:, None, :] - means_final[None, :, :], axis=2), axis=1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(4):
        axes[2].scatter(X[labels == k, 0], X[labels == k, 1], color=colors[k], alpha=0.5, s=25)
        axes[2].scatter([means_final[k, 0]], [means_final[k, 1]], color=colors[k], s=200, marker="X", edgecolors="black", lw=1.5, zorder=5)
    axes[2].set_title("(c) Converged K-Means ($K = 4$)\n(Hard Voronoi Decision Boundaries)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel(r"Feature Dimension 1 ($F_1$ Hz)")

    fig.tight_layout()
    _save(fig, "05_kmeans_clustering_lbg_splitting.png")


# -----------------------------------------------------------------------------
# Figure 6: Hard Clustering (K-Means) vs. Soft Probabilistic Modeling (GMM)
# -----------------------------------------------------------------------------
def plot_hard_vs_soft_clustering():
    np.random.seed(25)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Two overlapping 2D Gaussian clusters + 1 distinct outlier
    n1, n2 = 120, 120
    X1 = np.random.randn(n1, 2) * [0.9, 0.8] + [2.0, 3.0]
    X2 = np.random.randn(n2, 2) * [0.8, 1.1] + [5.0, 3.5]
    outlier = np.array([[8.5, 7.0]])
    X = np.vstack([X1, X2, outlier])

    # 1. K-Means Hard Boundaries
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    # Outlier is assigned to cluster 2
    d1 = np.linalg.norm(X - mu1, axis=1)
    d2 = np.linalg.norm(X - mu2, axis=1)
    assign = d1 > d2

    axes[0].scatter(X[~assign, 0], X[~assign, 1], color="teal", alpha=0.55, s=35, label="Cluster 1 (100% hard)")
    axes[0].scatter(X[assign, 0], X[assign, 1], color="darkorange", alpha=0.55, s=35, label="Cluster 2 (100% hard)")
    axes[0].scatter([outlier[0, 0]], [outlier[0, 1]], color="crimson", s=140, marker="*", zorder=6, label="Outlier point")
    axes[0].scatter([mu1[0]], [mu1[1]], color="teal", s=180, marker="X", edgecolors="k", lw=1.5, zorder=5)
    axes[0].scatter([mu2[0]], [mu2[1]], color="darkorange", s=180, marker="X", edgecolors="k", lw=1.5, zorder=5)
    axes[0].set_title("(a) K-Means Hard Clustering\nOutlier strictly pulls Cluster 2 centroid ($r_{ik} \\in \\{0, 1\\}$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Feature Dimension $x_1$")
    axes[0].set_ylabel("Feature Dimension $x_2$")
    axes[0].legend(loc="lower right", fontsize=8.5)

    # 2. GMM Soft Probabilistic Contours
    # Evaluate soft responsibilities for GMM
    def eval_normal(pt, mean, cov):
        d = len(mean)
        diff = pt - mean
        inv = np.linalg.inv(cov)
        return (1.0 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))) * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=-1))

    cov1 = np.array([[0.81, 0.1], [0.1, 0.64]])
    cov2 = np.array([[0.64, -0.1], [-0.1, 1.21]])
    p1 = 0.5 * eval_normal(X, mu1, cov1)
    p2 = 0.5 * eval_normal(X, mu2, cov2)
    gamma1 = p1 / (p1 + p2)

    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=gamma1, cmap="coolwarm", s=35, alpha=0.75, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=axes[1], shrink=0.8)
    cbar.set_label(r"Soft Responsibility $\gamma_{i1} = P(C_1 \mid x_i)$", fontsize=10)

    # Draw Gaussian 1-sigma, 2-sigma ellipses
    for scale, alpha in [(1, 0.4), (2, 0.2)]:
        e1 = Ellipse(xy=mu1, width=2 * scale * np.sqrt(0.81), height=2 * scale * np.sqrt(0.64), angle=10,
                     edgecolor="blue", fc="none", lw=1.8, linestyle="--")
        e2 = Ellipse(xy=mu2, width=2 * scale * np.sqrt(0.64), height=2 * scale * np.sqrt(1.21), angle=-10,
                     edgecolor="red", fc="none", lw=1.8, linestyle="--")
        axes[1].add_patch(e1)
        axes[1].add_patch(e2)

    axes[1].scatter([outlier[0, 0]], [outlier[0, 1]], color="purple", s=140, marker="*", zorder=6)
    axes[1].set_title("(b) GMM Soft Density Estimation (EM)\nContinuous responsibility & distributed influence ($\gamma_{ik} \\in [0, 1]$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Feature Dimension $x_1$")

    fig.tight_layout()
    _save(fig, "06_kmeans_hard_vs_gmm_soft.png")


# -----------------------------------------------------------------------------
# Figure 7: GMM 1D Likelihood Decomposition and Component Posteriors
# -----------------------------------------------------------------------------
def plot_gmm_1d_posteriors():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # IIT Madras Exam Specification: mu1=1, var1=1, w1=0.6; mu2=3, var2=1, w2=0.4
    mu1, var1, w1 = 1.0, 1.0, 0.6
    mu2, var2, w2 = 3.0, 1.0, 0.4

    x = np.linspace(-3.0, 7.0, 1000)

    def g(val, m, v):
        return (1.0 / np.sqrt(2 * np.pi * v)) * np.exp(-((val - m) ** 2) / (2 * v))

    comp1 = w1 * g(x, mu1, var1)
    comp2 = w2 * g(x, mu2, var2)
    composite = comp1 + comp2

    # Panel 1: Composite Likelihood
    axes[0].plot(x, comp1, color="teal", linestyle="--", lw=1.8, label=r"Weighted Component 1: $w_1 \mathcal{N}(1, 1) = 0.6 \mathcal{N}$")
    axes[0].plot(x, comp2, color="darkorange", linestyle="--", lw=1.8, label=r"Weighted Component 2: $w_2 \mathcal{N}(3, 1) = 0.4 \mathcal{N}$")
    axes[0].plot(x, composite, color="navy", lw=2.5, label=r"Composite GMM Density $p(x) = \sum w_k \mathcal{N}_k$")

    # Mark x = 2.0 evaluation point (standard exam problem)
    eval_x = 2.0
    val_y = w1 * g(eval_x, mu1, var1) + w2 * g(eval_x, mu2, var2)
    axes[0].plot([eval_x], [val_y], "ro", ms=8, zorder=5)
    axes[0].axvline(eval_x, color="crimson", linestyle=":", lw=1.5)
    axes[0].text(eval_x + 0.15, val_y + 0.02, f"Exam Point: $x = 2.0$\n$p(x) = {val_y:.3f}$", fontsize=9.5, fontweight="bold", color="crimson")
    axes[0].set_title(r"(a) 1D Gaussian Mixture Density Decomposition $p(x) = w_1 \mathcal{N}_1(x) + w_2 \mathcal{N}_2(x)$", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Posteriors
    post1 = comp1 / composite
    post2 = comp2 / composite
    axes[1].plot(x, post1, color="teal", lw=2.2, label=r"Posterior $P(C_1 \mid x) = \frac{w_1 \mathcal{N}_1}{p(x)}$")
    axes[1].plot(x, post2, color="darkorange", lw=2.2, label=r"Posterior $P(C_2 \mid x) = \frac{w_2 \mathcal{N}_2}{p(x)}$")
    axes[1].axvline(eval_x, color="crimson", linestyle=":", lw=1.5)
    axes[1].plot([eval_x], [0.6], "ro", ms=8, zorder=5)
    axes[1].text(eval_x + 0.15, 0.62, "At $x = 2.0$:\n$P(C_1 \mid x) = 0.60$\n$P(C_2 \mid x) = 0.40$", fontsize=9.5, fontweight="bold", color="crimson")
    axes[1].axhline(0.5, color="gray", linestyle="--", lw=1.0, alpha=0.7)
    axes[1].set_title(r"(b) Component Posterior Responsibilities $\gamma_k(x) = P(C_k \mid x)$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Acoustic Feature Value ($x$)")
    axes[1].set_ylabel("Posterior Probability")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="center right", fontsize=9)

    fig.tight_layout()
    _save(fig, "07_gmm_density_decomposition_and_posteriors.png")


# -----------------------------------------------------------------------------
# Figure 8: 3-State Left-to-Right Bakis HMM Topology and State Duration
# -----------------------------------------------------------------------------
def plot_hmm_topology_duration():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Bakis 3-State Diagram
    axes[0].set_xlim(-0.5, 4.5)
    axes[0].set_ylim(-1.5, 2.5)
    axes[0].axis("off")

    state_coords = [(0.5, 0.5), (2.0, 0.5), (3.5, 0.5)]
    state_names = ["State 1\n[Onset]", "State 2\n[Steady]", "State 3\n[Coda]"]
    colors = ["#2b5c8f", "#3e8e7e", "#c05c46"]

    for (x, y), name, col in zip(state_coords, state_names, colors):
        circle = plt.Circle((x, y), 0.42, color=col, ec="black", lw=2, zorder=3)
        axes[0].add_patch(circle)
        axes[0].text(x, y, name, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)

        # Self loop arrow
        axes[0].annotate("", xy=(x - 0.15, y + 0.42), xytext=(x + 0.15, y + 0.42),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-2.2", lw=1.8, color="black"))
        axes[0].text(x, y + 0.95, r"$a_{ii}$", ha="center", fontsize=10, fontweight="bold")

    # Forward arrows
    axes[0].annotate("", xy=(1.58, 0.5), xytext=(0.92, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    axes[0].text(1.25, 0.65, r"$a_{12}$", ha="center", fontsize=10, fontweight="bold")

    axes[0].annotate("", xy=(3.08, 0.5), xytext=(2.42, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    axes[0].text(2.75, 0.65, r"$a_{23}$", ha="center", fontsize=10, fontweight="bold")

    # Skip arrow from 1 to 3
    axes[0].annotate("", xy=(3.3, 0.15), xytext=(0.7, 0.15),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", lw=1.5, linestyle="--", color="gray"))
    axes[0].text(2.0, -0.65, r"$a_{13}$ (Skip transition)", ha="center", fontsize=9, color="gray")

    axes[0].set_title("(a) 3-State Left-to-Right (Bakis) HMM Topology", fontsize=11, fontweight="bold")

    # Panel 2: State Duration Distribution
    d = np.arange(1, 21)
    a_vals = [0.6, 0.75, 0.9]
    a_colors = ["darkgreen", "navy", "crimson"]

    for a_ii, col in zip(a_vals, a_colors):
        p_d = (a_ii ** (d - 1)) * (1.0 - a_ii)
        mean_d = 1.0 / (1.0 - a_ii)
        axes[1].plot(d, p_d, "-o", color=col, lw=2, ms=5,
                     label=rf"$a_{{ii}} = {a_ii}$ ($\mathbb{{E}}[d] = {mean_d:.1f}$ frames)")

    axes[1].set_title(r"(b) Geometric State Duration Distribution $P(d) = a_{ii}^{d-1}(1 - a_{ii})$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Duration in State ($d$ frames, 10 ms each)")
    axes[1].set_ylabel("Probability $P(d)$")
    axes[1].set_xticks(np.arange(1, 21, 2))
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, "08_hmm_bakis_topology_and_duration.png")


# -----------------------------------------------------------------------------
# Figure 9: Viterbi Trellis Dynamic Programming Search
# -----------------------------------------------------------------------------
def plot_viterbi_trellis():
    fig, ax = plt.subplots(figsize=(13, 6))

    n_states = 3
    n_times = 6
    state_names = [r"$S_1$ (Onset)", r"$S_2$ (Steady)", r"$S_3$ (Coda)"]

    # Pre-calculated optimal path for demonstration
    best_path = [0, 0, 1, 1, 2, 2]

    # Plot all trellis nodes
    for t in range(n_times):
        for s in range(n_states):
            is_best = (best_path[t] == s)
            col = "crimson" if is_best else "lightgray"
            edge_col = "darkred" if is_best else "gray"
            size = 180 if is_best else 100
            ax.scatter(t, s, color=col, s=size, edgecolors=edge_col, lw=1.5, zorder=4)

    # Plot candidate transitions (dashed) and survivor path (bold)
    for t in range(n_times - 1):
        for s_from in range(n_states):
            for s_to in [s_from, s_from + 1]:  # Left-to-right Bakis
                if s_to < n_states:
                    is_winner = (best_path[t] == s_from and best_path[t + 1] == s_to)
                    if is_winner:
                        ax.annotate("", xy=(t + 1, s_to), xytext=(t, s_from),
                                    arrowprops=dict(arrowstyle="->", lw=3.0, color="crimson"), zorder=5)
                    else:
                        ax.plot([t, t + 1], [s_from, s_to], color="lightgray", linestyle="--", lw=1.0, zorder=2)

    ax.set_xticks(range(n_times))
    ax.set_xticklabels([f"$t = {t+1}$\n($o_{t+1}$)" for t in range(n_times)], fontsize=10)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_names, fontsize=11, fontweight="bold")
    ax.set_title(r"Viterbi Dynamic Programming Trellis: Induction $\delta_t(j) = \max_i [\delta_{t-1}(i) a_{ij}] b_j(o_t)$", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frame ($t$)", fontsize=11)
    ax.set_ylabel("HMM State ($j$)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Annotate winner path
    ax.plot([], [], color="crimson", lw=3.0, label=r"Optimal Survivor Path $Q^* = (q_1^*, \dots, q_T^*)$")
    ax.plot([], [], color="lightgray", linestyle="--", lw=1.0, label="Pruned Competitor Paths")
    ax.legend(loc="upper left", fontsize=9.5)

    fig.tight_layout()
    _save(fig, "09_viterbi_trellis_decoding.png")


# -----------------------------------------------------------------------------
# Figure 10: Triphone Coarticulation and Decision Tree State Tying
# -----------------------------------------------------------------------------
def plot_triphones_and_senones():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Coarticulation Formant Shift for /ae/
    # Vowel /ae/ in "cat" (velar + alveolar), "bat" (bilabial + alveolar), "man" (nasal)
    contexts = [
        ("sil-k+ae (cat)", 750, 1550, "crimson"),
        ("sil-b+ae (bat)", 700, 1680, "teal"),
        ("sil-m+ae (man)", 820, 1420, "darkorange"),
        ("Neutral /ae/", 750, 1600, "navy"),
    ]

    for name, f1, f2, col in contexts:
        axes[0].scatter([f2], [f1], color=col, s=160, edgecolors="k", lw=1.2, label=name)
        # Confidence ellipse for each context
        e = Ellipse(xy=(f2, f1), width=120, height=60, angle=15, edgecolor=col, fc="none", lw=1.5, linestyle="--")
        axes[0].add_patch(e)

    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    axes[0].set_title(r"(a) Coarticulation: Acoustic Shift of /$\ae$/ Formants in Context", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(r"Second Formant $F_2$ (Hz) [Inverted standard]")
    axes[0].set_ylabel(r"First Formant $F_1$ (Hz) [Inverted standard]")
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Panel 2: Decision Tree State Tying (Senones)
    axes[1].set_xlim(-0.5, 4.5)
    axes[1].set_ylim(-0.5, 4.5)
    axes[1].axis("off")

    # Root node
    axes[1].text(2.0, 4.0, "Root: All Central States of /ae/\n(64,000 theoretical triphones)",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#e0eaf5", ec="#2b5c8f", lw=1.5),
                 fontsize=8.5, fontweight="bold")

    # Question 1: Left context nasal?
    axes[1].annotate("", xy=(1.0, 2.8), xytext=(1.8, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(1.1, 3.3, "Yes (Nasal)", fontsize=8, color="darkgreen", fontweight="bold")
    axes[1].annotate("", xy=(3.0, 2.8), xytext=(2.2, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(2.7, 3.3, "No", fontsize=8, color="crimson", fontweight="bold")

    # Child 1 (Nasal cluster)
    axes[1].text(1.0, 2.5, "Senone S_42\n[Tied /ae/ in Nasals]\ne.g., m-ae+n, n-ae+m",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", ec="darkgreen", lw=1.5),
                 fontsize=8)

    # Child 2: Question 2 (Right context stop?)
    axes[1].text(3.0, 2.5, "Right context stop?\n(/p/, /t/, /k/)",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#856404", lw=1.5),
                 fontsize=8)

    # Leaves from Q2
    axes[1].annotate("", xy=(2.3, 1.2), xytext=(2.8, 2.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(2.2, 1.7, "Yes", fontsize=8, color="darkgreen", fontweight="bold")
    axes[1].annotate("", xy=(3.7, 1.2), xytext=(3.2, 2.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(3.5, 1.7, "No", fontsize=8, color="crimson", fontweight="bold")

    axes[1].text(2.3, 0.8, "Senone S_88\n[Tied /ae/ before stops]\ne.g., k-ae+t, b-ae+t",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#cce5ff", ec="#004085", lw=1.5),
                 fontsize=8)
    axes[1].text(3.7, 0.8, "Senone S_104\n[Tied /ae/ elsewhere]\ne.g., s-ae+s, f-ae+l",
                 ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="#721c24", lw=1.5),
                 fontsize=8)

    axes[1].set_title(r"(b) Phonetic Decision Tree State Tying (Senones Clustering)", fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, "10_triphone_coarticulation_state_tying.png")


def main():
    print("Generating Week 2 Publication Plots...")
    plot_mfcc_pipeline()
    plot_mel_warping()
    plot_dct_decorrelation()
    plot_dynamic_deltas()
    plot_kmeans_lbg_splitting = plot_kmeans_lbg
    plot_kmeans_lbg_splitting()
    plot_hard_vs_soft_clustering()
    plot_gmm_1d_posteriors()
    plot_hmm_topology_duration()
    plot_viterbi_trellis()
    plot_triphones_and_senones()
    print("All 10 Week 2 plots generated successfully.")


if __name__ == "__main__":
    main()
