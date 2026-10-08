"""
Deterministic figure generation for Week 1 of Speech Technology.

Style: Seaborn whitegrid, talk context. NumPy and Matplotlib only.
Outputs PNGs to 'week 1/assets/'. Each function owns its own RNG seed.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Annulus, Circle

np.random.seed(0)  # freeze any library path that still touches the global RNG
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "sans-serif"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "week 1", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(ASSETS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Generated] {path}")


# -----------------------------------------------------------------------------
# Figure 1: Nyquist-Shannon sampling and aliasing
# -----------------------------------------------------------------------------
def plot_sampling_aliasing():
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    f_signal = 100.0
    t_continuous = np.linspace(0, 0.04, 2000)
    y_continuous = np.cos(2 * np.pi * f_signal * t_continuous)

    cases = [
        (2.6, r"$f_s / f = 2.6$ ($f_s = 260\ \mathrm{Hz} > 2f$) — Oversampled", "teal"),
        (2.0, r"$f_s / f = 2.0$ ($f_s = 200\ \mathrm{Hz} = 2f$) — Critical Nyquist rate", "darkgreen"),
        (1.4, r"$f_s / f = 1.4$ ($f_s = 140\ \mathrm{Hz} < 2f$) — Alias at $40\ \mathrm{Hz}$", "darkorange"),
        (0.8, r"$f_s / f = 0.8$ ($f_s = 80\ \mathrm{Hz} < 2f$) — Alias at $20\ \mathrm{Hz}$", "crimson"),
    ]

    for ax, (ratio, title, color) in zip(axes, cases):
        fs = ratio * f_signal
        t_samples = np.arange(0, 0.04 + 1e-9, 1.0 / fs)
        y_samples = np.cos(2 * np.pi * f_signal * t_samples)

        ax.plot(
            t_continuous * 1000,
            y_continuous,
            color="gray",
            linestyle="--",
            alpha=0.55,
            label="True analog 100 Hz cosine",
        )
        markerline, stemlines, _baseline = ax.stem(
            t_samples * 1000,
            y_samples,
            linefmt=color,
            markerfmt="o",
            basefmt="k-",
            label=f"Samples ($T_s = {1000 / fs:.2f}$ ms)",
        )
        plt.setp(markerline, color=color, markersize=6)
        plt.setp(stemlines, color=color, linewidth=1.5)

        if ratio < 2.0:
            folded = f_signal % fs
            if folded > fs / 2:
                folded = fs - folded
            y_alias = np.cos(2 * np.pi * folded * t_continuous)
            ax.plot(
                t_continuous * 1000,
                y_alias,
                color=color,
                linewidth=2,
                label=f"Principal alias ({folded:.0f} Hz)",
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.45, 1.55)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("Time (milliseconds)")
    fig.tight_layout()
    _save(fig, "01_sampling_aliasing.png")


# -----------------------------------------------------------------------------
# Figure 2: Uniform quantization and the 6.02 dB/bit rule
# -----------------------------------------------------------------------------
def _quantize_uniform(x, bits, xmax=1.0):
    """Mid-riser uniform quantizer on [-xmax, xmax] with 2**bits levels."""
    n_levels = 2**bits
    delta = 2 * xmax / n_levels
    k = np.floor((x + xmax) / delta)
    k = np.clip(k, 0, n_levels - 1)
    xq = -xmax + (k + 0.5) * delta
    return xq, delta


def plot_quantization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    t = np.linspace(0, 2.0, 2500)  # milliseconds
    x = np.sin(2 * np.pi * 1.0 * t)  # 1 kHz if t is in ms
    xq, _delta = _quantize_uniform(x, bits=3)
    axes[0].plot(t, x, color="gray", lw=1.6, label="Full-scale sinusoid")
    axes[0].step(t, xq, where="mid", color="crimson", lw=1.6, label="3-bit mid-riser quantizer")
    axes[0].set_title("Uniform quantization of a full-scale sinusoid", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time (milliseconds)")
    axes[0].set_ylabel("Amplitude (full scale)")
    axes[0].set_ylim(-1.25, 1.35)
    axes[0].legend(loc="upper right", fontsize=10)

    bits = np.arange(2, 17)
    # Long incommensurate sine so the error visits many points inside each bin.
    fs = 16000
    n = np.arange(fs)
    sine = np.sin(2 * np.pi * 997.0 * n / fs)
    measured = []
    for b in bits:
        xq_b, _delta_b = _quantize_uniform(sine, int(b))
        err = sine - xq_b
        measured.append(10 * np.log10(np.mean(sine**2) / np.mean(err**2)))
    measured = np.array(measured)
    theory = 10 * np.log10(1.5 * 2 ** (2 * bits))

    axes[1].plot(bits, theory, color="navy", lw=2.2, label=r"Theory $6.02B + 1.76$ dB")
    axes[1].plot(bits, measured, "o", color="crimson", ms=7, label="Measured full-scale sine")
    axes[1].set_title("Signal-to-quantization-noise ratio", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Word length $B$ (bits)")
    axes[1].set_ylabel("SQNR (dB)")
    axes[1].set_xticks(bits[::2])
    axes[1].legend(loc="upper left", fontsize=10)
    print(
        "[Quantization] B=16 measured "
        f"{measured[-1]:.3f} dB, theory {theory[-1]:.3f} dB"
    )

    fig.tight_layout()
    _save(fig, "02_quantization_sqnr.png")


# -----------------------------------------------------------------------------
# Figure 3: Fourier series of square, sawtooth, and triangle waves
# -----------------------------------------------------------------------------
def plot_fourier_decomposition():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.linspace(0, 2, 2000, endpoint=False)
    f0 = 1.0

    ax1 = axes[0, 0]
    square = np.sign(np.sin(2 * np.pi * f0 * t))
    ax1.plot(t, square, "k--", alpha=0.65, label="Ideal square wave")
    for harmonics in (1, 3, 7, 15):
        approx = np.zeros_like(t)
        for k in range(1, harmonics + 1, 2):
            approx += (4.0 / (np.pi * k)) * np.sin(2 * np.pi * k * f0 * t)
        ax1.plot(t, approx, lw=1.8 if harmonics == 15 else 1.2, label=f"Odd harmonics $k \\leq {harmonics}$")
    ax1.set_title(r"Square wave: odd harmonics, amplitude $\propto 1/k$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="lower right", fontsize=8)

    ax2 = axes[0, 1]
    saw = 2.0 * (t * f0 - np.floor(0.5 + t * f0))
    ax2.plot(t, saw, "k--", alpha=0.65, label="Ideal sawtooth wave")
    for harmonics in (1, 3, 7, 15):
        approx = np.zeros_like(t)
        for k in range(1, harmonics + 1):
            approx += -(2.0 / (np.pi * k)) * np.sin(2 * np.pi * k * f0 * t)
        ax2.plot(t, approx, lw=1.8 if harmonics == 15 else 1.2, label=f"Harmonics $k \\leq {harmonics}$")
    ax2.set_title(r"Sawtooth wave: all harmonics, amplitude $\propto 1/k$", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)

    ax3 = axes[1, 0]
    # Odd triangle matched to the sine series below: peak +1 at t = T0/4.
    triangle = (2.0 / np.pi) * np.arcsin(np.sin(2 * np.pi * f0 * t))
    ax3.plot(t, triangle, "k--", alpha=0.65, label="Ideal triangle wave")
    tri_ref = None
    for harmonics in (1, 3, 7):
        approx = np.zeros_like(t)
        for n_term, k in enumerate(range(1, harmonics + 1, 2)):
            approx += ((8.0 / (np.pi**2)) * ((-1) ** n_term) / (k**2)) * np.sin(2 * np.pi * k * f0 * t)
        tri_ref = approx
        ax3.plot(t, approx, lw=1.8 if harmonics == 7 else 1.2, label=f"Odd harmonics $k \\leq {harmonics}$")
    ax3.set_title(r"Triangle wave: odd harmonics, amplitude $\propto 1/k^{2}$", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"Normalized time ($t / T_0$)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="lower right", fontsize=8)
    # High-order check that the series and the drawn target are the same wave.
    tri_hi = np.zeros_like(t)
    for n_term, k in enumerate(range(1, 81, 2)):
        tri_hi += ((8.0 / (np.pi**2)) * ((-1) ** n_term) / (k**2)) * np.sin(2 * np.pi * k * f0 * t)
    print(f"[Triangle] max |series_{{k<=79}} - target| = {np.max(np.abs(tri_hi - triangle)):.4f}")
    _ = tri_ref

    ax4 = axes[1, 1]
    k_vals = np.arange(1, 16)
    saw_amp = 2.0 / (np.pi * k_vals)
    square_amp = np.array([4.0 / (np.pi * k) if k % 2 else 0.0 for k in k_vals])
    tri_amp = np.array([8.0 / (np.pi**2 * k**2) if k % 2 else 0.0 for k in k_vals])
    stems = (
        (k_vals - 0.22, saw_amp, "C0", "Sawtooth (every $k$)"),
        (k_vals, square_amp, "crimson", "Square (odd $k$)"),
        (k_vals + 0.22, tri_amp, "C2", "Triangle (odd $k$)"),
    )
    for xpos, amp, color, label in stems:
        markerline, stemlines, _baseline = ax4.stem(xpos, amp, linefmt="-", markerfmt="o", basefmt="k-")
        plt.setp(markerline, color=color, markersize=5)
        plt.setp(stemlines, color=color, linewidth=1.3)
        ax4.plot([], [], color=color, marker="o", lw=1.3, label=label)
    ax4.set_title(r"One-sided sine amplitudes $|b_k|$", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"Harmonic index $k$")
    ax4.set_ylabel(r"Sine-series amplitude $|b_k|$")
    ax4.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, "03_fourier_series_decomposition.png")


# -----------------------------------------------------------------------------
# Figure 4: DFT symmetry, leakage, and window sidelobes
# -----------------------------------------------------------------------------
def _first_sidelobes_db(window):
    """Return (first sidelobe dB, highest sidelobe dB) of a real window."""
    spectrum = np.abs(np.fft.rfft(window, n=32 * len(window)))
    db = 20 * np.log10(spectrum / spectrum.max() + 1e-30)
    slope = np.diff(db)
    maxima = np.where((slope[:-1] > 0) & (slope[1:] <= 0))[0] + 1
    maxima = maxima[maxima > 5]
    return float(db[maxima[0]]), float(np.max(db[maxima[:8]]))


def plot_dft_leakage():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fs = 8000
    n_dft = 64
    n = np.arange(n_dft)
    bin_hz = fs / n_dft
    k_axis = np.arange(n_dft)

    tone_on = np.cos(2 * np.pi * 8 * n / n_dft)  # exactly bin 8 = 1000 Hz
    tone_off = np.cos(2 * np.pi * 8.5 * n / n_dft)  # halfway, 1062.5 Hz
    mag_on = np.abs(np.fft.fft(tone_on))
    mag_off = np.abs(np.fft.fft(tone_off))
    mag_off_hamm = np.abs(np.fft.fft(tone_off * np.hamming(n_dft)))

    panels = (
        (axes[0, 0], mag_on, "teal", "On-bin cosine, rectangular window\n$f = 1000$ Hz $= 8 \\cdot f_s/N$"),
        (axes[0, 1], mag_off, "darkorange", "Half-bin detuned cosine, rectangular window\n$f = 1062.5$ Hz"),
        (axes[1, 0], mag_off_hamm, "purple", "Same detuned cosine, Hamming window\nLeakage into distant bins is suppressed"),
    )
    for ax, mag, color, title in panels:
        markerline, stemlines, _baseline = ax.stem(k_axis, mag, linefmt="-", markerfmt="o", basefmt="k-")
        plt.setp(markerline, color=color, markersize=4)
        plt.setp(stemlines, color=color, linewidth=1.2)
        ax.axvline(n_dft / 2, color="gray", ls=":", lw=1.2, label="Nyquist bin $N/2$")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"DFT bin $k$")
        ax.set_ylabel(r"Magnitude $|X[k]|$")
        ax.set_xlim(-1, n_dft)
        ax.legend(loc="upper right", fontsize=9)
    print(
        f"[DFT] on-bin |X[8]|={mag_on[8]:.2f}, |X[56]|={mag_on[56]:.2f}, "
        f"other max={np.max(np.delete(mag_on, [8, 56])):.2e}; bin spacing={bin_hz:.1f} Hz"
    )

    ax = axes[1, 1]
    n_win = 256
    windows = {
        "Rectangular": np.ones(n_win),
        "Hann": np.hanning(n_win),
        "Hamming": np.hamming(n_win),
    }
    colors = {"Rectangular": "C0", "Hann": "C2", "Hamming": "C3"}
    for name, window in windows.items():
        n_fft = 32 * n_win
        spectrum = np.abs(np.fft.rfft(window, n=n_fft))
        db = 20 * np.log10(spectrum / spectrum.max() + 1e-30)
        bins = np.fft.rfftfreq(n_fft, d=1.0 / n_win)
        ax.plot(bins, db, color=colors[name], lw=1.8, label=name)
        first_db, high_db = _first_sidelobes_db(window)
        print(f"[Window] {name}: first sidelobe {first_db:.2f} dB, highest of first lobes {high_db:.2f} dB")
    ax.axhline(-13.3, color="C0", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(-31.5, color="C2", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(-42.7, color="C3", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlim(0, 24)
    ax.set_ylim(-90, 5)
    ax.set_title("Window magnitude (dB, normalized to the main lobe)", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Frequency (bins of an $N$-point DFT)")
    ax.set_ylabel("Normalized magnitude (dB)")
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save(fig, "04_dft_window_leakage.png")


# -----------------------------------------------------------------------------
# Figure 5: LTI convolution and formant filtering
# -----------------------------------------------------------------------------
def plot_lti_convolution():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fs = 8000
    n = np.arange(0, 240)
    t = n / fs

    f_pitch = 150.0
    x_clean = np.zeros(len(n))
    x_clean[n % int(fs / f_pitch) == 0] = 1.0
    noise = 0.35 * np.cos(2 * np.pi * 2200 * t) + 0.12 * rng.standard_normal(len(n))
    x_in = x_clean + noise

    f_res = 600.0
    bandwidth = 100.0
    radius = np.exp(-np.pi * bandwidth / fs)
    theta = 2.0 * np.pi * f_res / fs
    a1 = -2.0 * radius * np.cos(theta)
    a2 = radius**2
    b0 = 1.0 - radius

    h_time = np.zeros(len(n))
    h_time[0] = b0
    h_time[1] = -a1 * h_time[0]
    for i in range(2, len(n)):
        h_time[i] = -a1 * h_time[i - 1] - a2 * h_time[i - 2]

    # Causal recurrence of the same resonator. Finite support, so the DFT
    # product is illustrative rather than an infinite-length identity.
    y_out = np.zeros(len(n))
    for i in range(len(n)):
        y_out[i] = b0 * x_in[i]
        if i >= 1:
            y_out[i] -= a1 * y_out[i - 1]
        if i >= 2:
            y_out[i] -= a2 * y_out[i - 2]

    w_eval = np.linspace(0, np.pi, 1024)
    freqs_hz = w_eval * fs / (2 * np.pi)
    z_eval = np.exp(1j * w_eval)
    h_eval = b0 / (1.0 + a1 * z_eval ** (-1) + a2 * z_eval ** (-2))
    freqs = np.fft.rfftfreq(1024, 1 / fs)

    axes[0, 0].plot(t * 1000, x_in, color="steelblue", lw=1.1)
    axes[0, 0].set_title(r"Input $x[n]$: glottal pulses + 2200 Hz tone", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(freqs, 20 * np.log10(np.abs(np.fft.rfft(x_in, 1024)) + 1e-6), color="steelblue", lw=1.1)
    axes[0, 1].set_title(r"Input magnitude $|X[k]|$ (dB)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].set_xlim(0, 4000)

    axes[1, 0].plot(t[:80] * 1000, h_time[:80], color="crimson", lw=1.8)
    axes[1, 0].set_title(r"Impulse response $h[n]$ of a 600 Hz resonator", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Amplitude")

    axes[1, 1].plot(freqs_hz, 20 * np.log10(np.abs(h_eval) + 1e-6), color="crimson", lw=2.2)
    axes[1, 1].set_title(r"Resonator $|H(e^{j\omega})|$, pole angle at 600 Hz", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Gain (dB)")
    axes[1, 1].set_xlim(0, 4000)

    axes[2, 0].plot(t * 1000, y_out, color="forestgreen", lw=1.4)
    axes[2, 0].set_title(r"Output $y[n]$ from the causal recurrence", fontsize=12, fontweight="bold")
    axes[2, 0].set_xlabel("Time (milliseconds)")
    axes[2, 0].set_ylabel("Amplitude")

    axes[2, 1].plot(freqs, 20 * np.log10(np.abs(np.fft.rfft(y_out, 1024)) + 1e-6), color="forestgreen", lw=1.3)
    axes[2, 1].set_title(r"Output $|Y[k]|$: 2200 Hz tone attenuated, 600 Hz kept", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("Magnitude (dB)")
    axes[2, 1].set_xlim(0, 4000)

    fig.tight_layout()
    _save(fig, "05_lti_convolution_filtering.png")


# -----------------------------------------------------------------------------
# Figure 6: Z-plane poles, causal ROC, and the formant magnitude
# -----------------------------------------------------------------------------
def plot_pole_zero_constellation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.4))
    theta = np.linspace(0, 2 * np.pi, 600)
    fs = 8000
    r1, theta1 = 0.92, 2 * np.pi * 500 / fs
    r2, theta2 = 0.88, 2 * np.pi * 1500 / fs
    poles = [
        r1 * np.exp(1j * theta1),
        r1 * np.exp(-1j * theta1),
        r2 * np.exp(1j * theta2),
        r2 * np.exp(-1j * theta2),
    ]
    zeros = [0.95, -0.7]
    max_pole_r = max(abs(p) for p in poles)

    # Causal ROC is the EXTERIOR of the outermost pole, not the disk inside it.
    ax1.add_patch(
        Annulus((0, 0), 2.0, 2.0 - max_pole_r, facecolor="teal", edgecolor="none", alpha=0.18, zorder=0)
    )
    ax1.plot(np.cos(theta), np.sin(theta), "k--", lw=1.4, label="Unit circle $|z| = 1$", zorder=2)
    ax1.plot(
        max_pole_r * np.cos(theta),
        max_pole_r * np.sin(theta),
        color="teal",
        lw=1.5,
        label=rf"Causal ROC boundary $|z| = {max_pole_r:.2f}$",
        zorder=2,
    )
    ax1.axhline(0, color="gray", lw=0.7, alpha=0.5)
    ax1.axvline(0, color="gray", lw=0.7, alpha=0.5)

    for pole in poles:
        ax1.plot(pole.real, pole.imag, "rx", ms=11, mew=2.4, zorder=4)
    ax1.plot([], [], "rx", ms=11, mew=2.4, label="Formant poles")
    for zero in zeros:
        ax1.plot(zero, 0.0, "o", ms=10, mfc="none", mec="C0", mew=2.0, zorder=4)
    ax1.plot([], [], "o", ms=10, mfc="none", mec="C0", mew=2.0, label="Zeros")

    ax1.set_xlim(-1.45, 1.45)
    ax1.set_ylim(-1.45, 1.45)
    ax1.set_aspect("equal")
    ax1.set_title("Causal ROC is outside the outermost pole\nShading is $|z| > 0.92$; the unit circle lies inside it", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Real part $\mathrm{Re}\{z\}$")
    ax1.set_ylabel(r"Imaginary part $\mathrm{Im}\{z\}$")
    ax1.legend(loc="upper right", fontsize=9)

    w = np.linspace(0, np.pi, 1200)
    freqs_hz = w * fs / (2 * np.pi)
    z_eval = np.exp(1j * w)
    numer = np.ones_like(z_eval)
    for zero in zeros:
        numer *= z_eval - zero
    denom = np.ones_like(z_eval)
    for pole in poles:
        denom *= z_eval - pole
    gain_db = 20 * np.log10(np.abs(numer / denom))
    ax2.plot(freqs_hz, gain_db, color="darkmagenta", lw=2.2)
    ax2.axvline(500, color="r", ls=":", label="Pole angle $F_1 = 500$ Hz")
    ax2.axvline(1500, color="darkorange", ls=":", label="Pole angle $F_2 = 1500$ Hz")
    ax2.set_title(r"$|H(e^{j\omega})|$ on the unit circle" "\nPeaks sit near the pole angles and are pulled by the zeros", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Gain (dB)")
    ax2.set_xlim(0, 4000)
    ax2.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save(fig, "06_pole_zero_constellation.png")


# -----------------------------------------------------------------------------
# Figure 7: Autocorrelation pitch estimation
# -----------------------------------------------------------------------------
def plot_autocorrelation_pitch():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fs = 16000
    t = np.arange(int(0.05 * fs)) / fs
    n_samples = len(t)

    f0_true = 125.0
    period = int(fs / f0_true)
    pulses = np.zeros(n_samples)
    pulses[::period] = 1.0
    resonator = np.exp(-t * 200) * (np.sin(2 * np.pi * 700 * t) + 0.6 * np.sin(2 * np.pi * 1220 * t))
    voiced = np.convolve(pulses, resonator)[:n_samples]
    voiced += 0.03 * rng.standard_normal(n_samples)
    unvoiced = np.diff(rng.standard_normal(n_samples), prepend=0.0)

    def normalized_autocorr(x):
        corr = np.correlate(x, x, mode="full")
        corr = corr[corr.size // 2 :]
        return corr / corr[0]

    r_voiced = normalized_autocorr(voiced)
    r_unvoiced = normalized_autocorr(unvoiced)
    lags_ms = np.arange(len(r_voiced)) / fs * 1000

    l_min = int(np.floor(fs / 500))
    l_max = int(np.ceil(fs / 50))
    l_star = l_min + int(np.argmax(r_voiced[l_min : l_max + 1]))
    print(
        f"[Pitch] l*={l_star} samples, T0={1000 * l_star / fs:.2f} ms, "
        f"F0={fs / l_star:.2f} Hz, rho={r_voiced[l_star]:.3f}; "
        f"unvoiced max in search={np.max(r_unvoiced[l_min:l_max + 1]):.3f}"
    )

    axes[0, 0].plot(t * 1000, voiced, color="navy", lw=1.2)
    axes[0, 0].set_title(r"Voiced frame (/a/-like, $F_0 = 125$ Hz)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude")

    axes[1, 0].plot(lags_ms[:400], r_voiced[:400], color="navy", lw=1.6)
    axes[1, 0].axvspan(1000 * l_min / fs, 1000 * l_max / fs, color="green", alpha=0.12, label="Search 50–500 Hz")
    axes[1, 0].axvline(1000 * l_star / fs, color="crimson", ls="--", lw=1.5, label=rf"Peak at {1000 * l_star / fs:.1f} ms")
    axes[1, 0].set_title(r"Voiced $\rho_{xx}[l]$: pitch peak inside the search band", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Lag (milliseconds)")
    axes[1, 0].set_ylabel("Normalized autocorrelation")
    axes[1, 0].legend(loc="upper right", fontsize=9)

    axes[0, 1].plot(t * 1000, unvoiced, color="darkslategray", lw=0.9)
    axes[0, 1].set_title("Unvoiced frame (differentiated noise, /s/-like)", fontsize=12, fontweight="bold")

    axes[1, 1].plot(lags_ms[:400], r_unvoiced[:400], color="darkslategray", lw=1.3)
    axes[1, 1].axvspan(1000 * l_min / fs, 1000 * l_max / fs, color="green", alpha=0.12)
    axes[1, 1].axhline(0.30, color="crimson", ls=":", lw=1.2, label=r"Voicing threshold $\rho = 0.30$")
    axes[1, 1].set_title(r"Unvoiced $\rho_{xx}[l]$: no peak above threshold", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Lag (milliseconds)")
    axes[1, 1].set_ylim(-0.45, 1.05)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, "07_autocorrelation_pitch.png")


# -----------------------------------------------------------------------------
# Figure 8: Wideband vs narrowband spectrograms
# -----------------------------------------------------------------------------
def plot_spectrogram_tradeoff():
    rng = np.random.default_rng(11)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fs = 16000
    duration = 0.8
    t = np.arange(int(fs * duration)) / fs

    f0 = 130 + 30 * np.sin(2 * np.pi * 2.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    glottal = (1.0 + np.sin(phase)) * (phase % (2 * np.pi) < 0.3)
    f1 = 450 + 300 * (t / duration)
    f2 = 1400.0
    speech = np.zeros_like(t)
    pulse_idx = np.flatnonzero(glottal > 0.8)
    for i in pulse_idx:
        remain = len(t) - i
        tau = np.arange(remain) / fs
        burst = np.exp(-tau * 250) * (np.sin(2 * np.pi * f1[i] * tau) + 0.5 * np.sin(2 * np.pi * f2 * tau))
        speech[i:] += burst
    speech += 0.02 * rng.standard_normal(len(speech))

    def stft_power(x, nperseg, noverlap):
        step = nperseg - noverlap
        n_frames = (len(x) - nperseg) // step
        window = np.hamming(nperseg)
        spec = np.zeros((nperseg // 2 + 1, n_frames))
        t_axis = (np.arange(n_frames) * step + nperseg / 2) / fs
        f_axis = np.fft.rfftfreq(nperseg, 1 / fs)
        for idx in range(n_frames):
            frame = x[idx * step : idx * step + nperseg] * window
            spec[:, idx] = np.abs(np.fft.rfft(frame)) ** 2
        return f_axis, t_axis, spec

    f_w, t_w, s_w = stft_power(speech, int(0.006 * fs), int(0.006 * fs * 0.75))
    f_n, t_n, s_n = stft_power(speech, int(0.050 * fs), int(0.050 * fs * 0.8))
    db_w = 10 * np.log10(s_w + 1e-6)
    db_n = 10 * np.log10(s_n + 1e-6)

    im1 = ax1.pcolormesh(t_w * 1000, f_w, db_w, shading="gouraud", cmap="viridis")
    ax1.set_ylim(0, 3500)
    ax1.set_title("Wideband spectrogram (6 ms Hamming window)\nVertical striations are individual glottal pulses", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=ax1, label="Power (dB, this panel)")

    im2 = ax2.pcolormesh(t_n * 1000, f_n, db_n, shading="gouraud", cmap="magma")
    ax2.set_ylim(0, 3500)
    ax2.set_title("Narrowband spectrogram (50 ms Hamming window)\nHorizontal lines are harmonics; $F_1$ glides upward", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (milliseconds)")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=ax2, label="Power (dB, this panel)")

    fig.tight_layout()
    _save(fig, "08_wideband_vs_narrowband_spectrogram.png")


# -----------------------------------------------------------------------------
# Figure 9: Source-filter model and real-cepstrum liftering
# -----------------------------------------------------------------------------
def plot_source_filter_cepstrum():
    fig, axes = plt.subplots(4, 1, figsize=(13, 12.5))
    fs = 16000
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    def formant_peak(f, center, bw, gain):
        return gain / (1.0 + ((f - center) / (bw / 2.0)) ** 2)

    v_mag = (
        formant_peak(freqs, 600, 100, 15)
        + formant_peak(freqs, 1400, 150, 10)
        + formant_peak(freqs, 2400, 200, 6)
        + 1.0
    )
    f0 = 125.0
    excitation = 0.05 * np.ones_like(freqs)
    for harmonic in np.arange(f0, fs / 2, f0):
        excitation += np.exp(-0.5 * ((freqs - harmonic) / 12.0) ** 2)
    excitation /= np.max(excitation[freqs < 400])
    speech = excitation * v_mag
    log_v = 20 * np.log10(v_mag)
    log_s = 20 * np.log10(speech)

    cepstrum = np.fft.irfft(log_s)
    quefrency_ms = np.arange(len(cepstrum)) / fs * 1000
    cutoff = int(0.003 * fs)  # 3 ms
    lifter = np.zeros_like(cepstrum)
    lifter[:cutoff] = 1.0
    lifter[-cutoff:] = 1.0
    recovered = np.fft.rfft(cepstrum * lifter).real

    axes[0].plot(freqs, speech, color="steelblue", lw=1.1, label=r"$|S| = |E|\,|V|$")
    axes[0].plot(freqs, v_mag, color="crimson", lw=2.0, label=r"Vocal-tract envelope $|V|$")
    axes[0].set_title(r"Harmonic speech spectrum under a formant envelope", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Linear amplitude")
    axes[0].set_xlim(0, 4000)
    axes[0].legend(loc="upper right", fontsize=10)

    axes[1].plot(freqs, log_s, color="darkslategray", lw=1.0, alpha=0.85, label=r"$\log|S(\omega)|$")
    axes[1].plot(freqs, log_v, color="crimson", lw=2.0, ls="--", label=r"$\log|V(\omega)|$ through the harmonic peaks")
    axes[1].set_title(r"Log spectrum: $\log|S| = \log|E| + \log|V|$  (radiation absorbed into $E$)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_xlim(0, 4000)
    axes[1].legend(loc="upper right", fontsize=10)

    axes[2].plot(quefrency_ms[:200], cepstrum[:200], color="purple", lw=1.5)
    axes[2].axvspan(0, 3.0, color="crimson", alpha=0.15, label="Lifter kept: $|n| < 3$ ms")
    axes[2].axvline(1000 / f0, color="C0", ls=":", lw=2.0, label=rf"Pitch rahmonic at {1000 / f0:.1f} ms")
    axes[2].set_title(r"Real cepstrum $\mathrm{IDFT}\{\log|S|\}$", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Quefrency (milliseconds)")
    axes[2].set_ylabel("Cepstral amplitude")
    axes[2].set_xlim(0, 20)
    axes[2].legend(loc="upper right", fontsize=10)

    axes[3].plot(freqs, log_s, color="0.75", lw=1.0, label="Raw log spectrum")
    axes[3].plot(freqs, log_v, color="crimson", lw=1.5, ls="--", label=r"True $\log|V|$")
    axes[3].plot(freqs, recovered, color="forestgreen", lw=2.3, label="3 ms lifter (same shape, lower level)")
    axes[3].set_title("Liftering removes the pitch rahmonic; valleys pull the level down", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("Log magnitude (dB)")
    axes[3].set_xlim(0, 4000)
    axes[3].legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    _save(fig, "09_source_filter_cepstral_separation.png")


# -----------------------------------------------------------------------------
# Figure 10: Mel scale and a 26-channel triangular filter bank
# -----------------------------------------------------------------------------
def plot_mel_filterbank():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
    fs = 16000
    n_filters = 26
    n_fft = 1024

    def hz_to_mel(hz):
        return 1125.0 * np.log(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (np.exp(mel / 1125.0) - 1.0)

    hz_range = np.linspace(0, 8000, 1200)
    ax1.plot(hz_range, hz_to_mel(hz_range), color="navy", lw=2.4, label=r"$1125\ln(1+f/700)$")
    chord = hz_range * (hz_to_mel(1000) / 1000)
    ax1.plot(hz_range, chord, "k--", alpha=0.7, label="Chord through 0 Hz and 1000 Hz")
    ax1.axvline(1000, color="crimson", ls=":", label="1000 Hz reference")
    ax1.set_title(r"Mel map used in this note: $f_{\mathrm{mel}} = 1125\ln(1 + f_{\mathrm{Hz}}/700)$", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Physical frequency (Hz)")
    ax1.set_ylabel("Perceived pitch (mels)")
    ax1.legend(loc="upper left", fontsize=10)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(fs / 2), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    freq_bins = np.fft.rfftfreq(n_fft, 1 / fs)
    colors = sns.color_palette("rocket", n_filters)
    for m in range(1, n_filters + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        weights = np.zeros(len(freq_bins))
        for k in range(left, center):
            weights[k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            weights[k] = (right - k) / max(1, right - center)
        ax2.plot(freq_bins, weights, color=colors[m - 1], lw=1.5)

    ax2.set_title(
        "26 triangular filters, uniform in mel, drawn so each triangle is visible\n"
        "The end-to-end lab uses 40 filters; the construction is the same",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Physical frequency (Hz)")
    ax2.set_ylabel("Filter weight")
    ax2.set_xlim(0, 8000)
    ax2.set_ylim(-0.02, 1.08)

    fig.tight_layout()
    _save(fig, "10_mel_filterbank_frequency_response.png")


if __name__ == "__main__":
    print("[Starting] Week 1 figure generation")
    plot_sampling_aliasing()
    plot_quantization()
    plot_fourier_decomposition()
    plot_dft_leakage()
    plot_lti_convolution()
    plot_pole_zero_constellation()
    plot_autocorrelation_pitch()
    plot_spectrogram_tradeoff()
    plot_source_filter_cepstrum()
    plot_mel_filterbank()
    print("[Completed] 10 figures")
