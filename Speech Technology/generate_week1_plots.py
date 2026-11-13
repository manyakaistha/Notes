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
# Figure 4: DFT finite-record leakage and window sidelobes
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
# Figure 5: LTI convolution and resonator filtering
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
    axes[0, 0].set_title(r"Input $x[n]$: periodic impulse train + 2200 Hz tone", fontsize=12, fontweight="bold")
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
    axes[2, 1].set_title(r"Output $|Y[k]|$: input spectrum shaped by resonator", fontsize=12, fontweight="bold")
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
# Figure 7: Synthetic speech classes and the acoustic waveform
# -----------------------------------------------------------------------------
def plot_speech_waveforms():
    fs = 16000
    t = np.arange(int(0.04 * fs)) / fs
    voiced = np.sin(2 * np.pi * 120 * t) + 0.35 * np.sin(2 * np.pi * 240 * t)
    rng = np.random.default_rng(42)
    unvoiced = 0.45 * rng.standard_normal(len(t))
    silence = np.zeros_like(t)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, x, title, color in zip(
        axes, (voiced, unvoiced, silence),
        ("Voiced: quasi-periodic excitation", "Unvoiced: aperiodic turbulent excitation", "Silence: near-zero acoustic pressure variation"),
        ("C0", "C1", "C2")):
        ax.plot(t * 1000, x, color=color, lw=1.1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative amplitude")
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    _save(fig, "07_speech_waveform_classes.png")


if __name__ == "__main__":
    print("[Starting] Week 1 in-scope figure generation")
    plot_sampling_aliasing()
    plot_quantization()
    plot_fourier_decomposition()
    plot_dft_leakage()
    plot_lti_convolution()
    plot_pole_zero_constellation()
    plot_speech_waveforms()
    print("[Completed] 7 in-scope figures")
