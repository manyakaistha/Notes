"""Reproducible Week 1 lab: sampled signals, transforms, systems, speech classes."""
import numpy as np


def main():
    # Sampling of a 1 kHz sinusoid at 8 kHz: no aliasing, 8 samples/cycle.
    fs, f0 = 8000, 1000
    n = np.arange(16)
    x = np.cos(2 * np.pi * f0 * n / fs)
    print(f"[Sampling] fs={fs} Hz, f0={f0} Hz, samples/cycle={fs // f0}")

    # Mid-rise uniform quantizer on [-1, 1), 8 bits.
    bits, xmax = 8, 1.0
    delta = 2 * xmax / (2**bits)
    q = -xmax + (np.floor((x + xmax) / delta) + 0.5) * delta
    q = np.clip(q, -xmax + delta / 2, xmax - delta / 2)
    print(f"[Quantization] bits={bits}, step={delta:.6f}, max_abs_error={np.max(np.abs(x-q)):.6f}")

    # Complex Fourier series of a periodic unit-amplitude square wave; first five odd terms.
    t = np.arange(4000) / 4000
    approx = np.zeros_like(t)
    for k in (1, 3, 5, 7, 9):
        approx += 4 / (np.pi * k) * np.sin(2 * np.pi * k * t)
    target = np.where(np.sin(2*np.pi*t) >= 0, 1.0, -1.0)
    print(f"[Fourier series] terms=5 odd harmonics, RMS_error={np.sqrt(np.mean((target-approx)**2)):.6f}")

    # DFT bin spacing and reconstruction via inverse DFT for a bin-centered tone.
    N, fs_dft = 64, 8000
    nd = np.arange(N)
    tone = np.cos(2*np.pi*8*nd/N)
    X = np.fft.fft(tone)
    rec_error = np.max(np.abs(np.fft.ifft(X).real - tone))
    print(f"[DFT] N={N}, bin_spacing={fs_dft/N:.1f} Hz, peak_bins=8,56, inverse_max_error={rec_error:.2e}")

    # Finite discrete convolution demonstrates LTI superposition mechanism.
    a = np.array([1., 2., 0.])
    h = np.array([1., -0.5])
    y = np.convolve(a, h)
    print(f"[Convolution] x={a.tolist()}, h={h.tolist()}, y={y.tolist()}")

    # Basic acoustic labels: periodic voiced, noise-like unvoiced, silence.
    rng = np.random.default_rng(42)
    t = np.arange(320) / 16000
    voiced = np.sin(2*np.pi*120*t) + 0.35*np.sin(2*np.pi*240*t)
    unvoiced = 0.45*rng.standard_normal(t.size)
    silence = np.zeros_like(t)
    for name, signal in (("voiced", voiced), ("unvoiced", unvoiced), ("silence", silence)):
        print(f"[Waveform] {name}: rms={np.sqrt(np.mean(signal**2)):.6f}")


if __name__ == "__main__":
    main()
