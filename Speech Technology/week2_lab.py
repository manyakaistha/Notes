"""Reproducible numeric exercises for the screenshot-bounded Week 2 syllabus."""
import math
import numpy as np


def main():
    # Physical-analysis values drawn from the stated 16 kHz / 25 ms / 10 ms setup.
    fs, frame_ms, hop_ms = 16000, 25, 10
    frame_n, hop_n = round(fs*frame_ms/1000), round(fs*hop_ms/1000)
    print(f"Frame samples: {frame_n}; hop samples: {hop_n}; overlap samples: {frame_n-hop_n}")

    mel_1000 = 1127*np.log1p(1000/700)
    print(f"Mel(1000 Hz): {mel_1000:.2f} mels")

    # Four-point orthonormal DCT-II and exact full-coefficient inverse check.
    x=np.array([2.,1.,0.,-1.])
    k, n = np.arange(4)[:,None], np.arange(4)[None,:]
    D=np.sqrt(2/4)*np.cos(np.pi*k*(n+.5)/4)
    D[0] /= np.sqrt(2)
    c=D @ x
    xr=D.T @ c
    print("DCT-II coefficients: " + np.array2string(c, precision=4))
    print(f"DCT reconstruction max error: {np.max(np.abs(x-xr)):.2e}")

    # Gaussian MLE uses denominator N for maximum likelihood (not unbiased N-1).
    obs=np.array([1.,2.,3.,4.])
    mu=obs.mean(); var=np.mean((obs-mu)**2)
    print(f"Gaussian MLE: mean={mu:.2f}, variance={var:.2f}, std={np.sqrt(var):.2f}")
    # PDF values have inverse-feature units and are not point probabilities.
    def pdf(value, mean, variance):
        return np.exp(-.5*(value-mean)**2/variance)/np.sqrt(2*np.pi*variance)
    p0,p1=pdf(1.,0.,1.),pdf(1.,2.,1.)
    print(f"At x=1: density N(0,1)={p0:.6f}; density N(2,1)={p1:.6f}")

    # Exact Bayes error for equal priors/common variance: threshold midpoint.
    threshold=1.
    print(f"Equal-prior decision threshold for N(0,1) vs N(2,1): {threshold:.2f}")
    error = 0.5*math.erfc(1/math.sqrt(2))
    print(f"Equal-prior error: {error:.6f} (standard-normal tail beyond 1)")

    # Complete synthetic front end: signal -> overlapping frames -> power FFT ->
    # triangular Mel energies -> log -> orthonormal DCT-II (13/26 coefficients).
    fs=16000
    time=np.arange(int(.12*fs))/fs
    signal=.7*np.sin(2*np.pi*125*time)+.25*np.sin(2*np.pi*700*time)+.12*np.sin(2*np.pi*1500*time)
    frame_n, hop_n, nfft, nmel = 400, 160, 512, 26
    frames=np.stack([signal[s:s+frame_n] for s in range(0,len(signal)-frame_n+1,hop_n)])
    window=np.hamming(frame_n)
    power=np.abs(np.fft.rfft(frames*window,n=nfft,axis=1))**2
    hz=np.fft.rfftfreq(nfft,1/fs)
    mel=lambda f:1127*np.log1p(f/700)
    inv=lambda m:700*np.expm1(m/1127)
    edges=inv(np.linspace(mel(0),mel(fs/2),nmel+2))
    bank=np.zeros((nmel,len(hz)))
    for j in range(nmel):
        left,center,right=edges[j:j+3]
        bank[j]=np.maximum(0,np.minimum((hz-left)/(center-left),(right-hz)/(right-center)))
    log_energy=np.log(np.maximum(np.einsum("tf,mf->tm",power,bank),1e-12))
    k,m=np.arange(13)[:,None],np.arange(nmel)[None,:]
    dct=np.sqrt(2/nmel)*np.cos(np.pi*k*(m+.5)/nmel)
    dct[0]/=np.sqrt(2)
    mfcc=np.einsum("tm,km->tk",log_energy,dct)
    print(f"Synthetic front end: frames={frames.shape}, Mel energies={log_energy.shape}, MFCC={mfcc.shape}")
    print(f"First-frame MFCC[0:5]: {np.array2string(mfcc[0,:5], precision=3)}")
    print(f"MFCC finite: {bool(np.isfinite(mfcc).all())}")


if __name__ == "__main__": main()
