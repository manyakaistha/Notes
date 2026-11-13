"""Deterministic discrete-HMM and HMM-GMM laboratory for Week 4."""
from itertools import product
import math
import numpy as np

STATES = ("Sunny", "Cloudy", "Rainy")
SYMBOLS = ("Dry", "Dryish", "Damp", "Soggy")
A = np.array([[0.500, 0.375, 0.125],
              [0.250, 0.125, 0.625],
              [0.250, 0.075, 0.675]], dtype=float)
PI = A[1].copy()  # Day 0 is Cloudy; transition once to the first emitting state.
SPEECH_STATES = ("Acoustic 1", "Acoustic 2", "Acoustic 3")
SPEECH_PI = np.array([1., 0., 0.])
SPEECH_A = np.array([[.6, .4, 0.], [0., .5, .5], [0., 0., 1.]])
B = np.array([[0.60, 0.20, 0.15, 0.05],
              [0.25, 0.25, 0.25, 0.25],
              [0.05, 0.10, 0.35, 0.50]], dtype=float)
OBS = (0, 0, 3)  # Dry, Dry, Soggy

def logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(x - m).sum()))

def logp(x):
    values = np.asarray(x, dtype=float)
    result = np.full(values.shape, -np.inf)
    np.log(values, out=result, where=values > 0)
    return result

def viterbi(pi, a, b, obs):
    n, tmax = len(pi), len(obs)
    d = np.full((tmax, n), -np.inf)
    back = np.full((tmax, n), -1, dtype=int)
    d[0] = logp(pi) + logp(b[:, obs[0]])
    for t in range(1, tmax):
        for j in range(n):
            candidates = d[t - 1] + logp(a[:, j])
            back[t, j] = int(np.argmax(candidates))
            d[t, j] = candidates[back[t, j]] + logp(b[j, obs[t]])
    if not np.isfinite(d[-1].max()):
        raise ValueError("No supported state path for these observations")
    path = np.empty(tmax, dtype=int)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(tmax - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, d, back, float(np.max(d[-1]))

def forward_backward(pi, a, b, obs):
    n, tmax = len(pi), len(obs)
    la, lb = logp(a), logp(b)
    alpha = np.full((tmax, n), -np.inf)
    beta = np.full((tmax, n), -np.inf)
    alpha[0] = logp(pi) + lb[:, obs[0]]
    for t in range(1, tmax):
        for j in range(n):
            alpha[t, j] = lb[j, obs[t]] + logsumexp(alpha[t-1] + la[:, j])
    beta[-1] = 0.0
    for t in range(tmax-2, -1, -1):
        for i in range(n):
            beta[t, i] = logsumexp(la[i] + lb[:, obs[t+1]] + beta[t+1])
    ll = logsumexp(alpha[-1])
    if not np.isfinite(ll):
        raise ValueError("Zero observation evidence: posterior is undefined")
    gamma = np.exp(alpha + beta - ll)
    xi = np.empty((tmax-1, n, n))
    for t in range(tmax-1):
        xi[t] = np.exp(alpha[t, :, None] + la + lb[None, :, obs[t+1]] + beta[t+1] - ll)
    filt = np.exp(alpha - np.array([logsumexp(row) for row in alpha])[:, None])
    return alpha, beta, gamma, xi, filt, ll

def enumerate_paths(pi, a, b, obs):
    records = []
    for path in product(range(len(pi)), repeat=len(obs)):
        p = pi[path[0]] * b[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= a[path[t-1], path[t]] * b[path[t], obs[t]]
        records.append((path, p))
    return records

def gaussian_mixture_pdf(x, weights, means, variances):
    x = np.asarray(x, dtype=float)
    density = np.zeros_like(x)
    for w, mu, var in zip(weights, means, variances):
        density += w * np.exp(-0.5*(x-mu)**2/var) / math.sqrt(2*math.pi*var)
    return density

def hmm_gmm_demo():
    # Explicit state-conditional 1-D two-component GMMs (feature units: normalized log-energy).
    weights = np.array([[.65,.35],[.55,.45],[.40,.60]])
    means = np.array([[-1.2,-.2],[-.3,.6],[.5,1.3]])
    variances = np.array([[.20,.12],[.25,.18],[.22,.16]])
    features = np.array([-1.05,-.48,-.10,.42,.82,1.08])
    g = np.stack([gaussian_mixture_pdf(features, weights[s], means[s], variances[s]) for s in range(3)])
    path, delta, back, best = viterbi(SPEECH_PI, SPEECH_A, g, range(len(features)))
    _, _, gamma, xi, filt, ll = forward_backward(SPEECH_PI, SPEECH_A, g, range(len(features)))
    assert np.allclose(weights.sum(1), 1) and np.all(variances > 0)
    assert np.allclose(SPEECH_A.sum(1), 1)
    checks = enumerate_paths(SPEECH_PI, SPEECH_A, g, range(len(features)))
    assert np.isclose(sum(v for _, v in checks), math.exp(ll))
    assert np.isclose(max(v for _, v in checks), math.exp(best))
    assert np.allclose(gamma.sum(1), 1) and np.allclose(xi.sum((1,2)), 1)
    return weights, means, variances, features, g, path, delta, back, best, gamma, xi, filt, ll

def run():
    path, delta, back, best = viterbi(PI, A, B, OBS)
    alpha, beta, gamma, xi, filt, ll = forward_backward(PI, A, B, OBS)
    records = enumerate_paths(PI, A, B, OBS)
    total = sum(p for _, p in records)
    max_enum = max(p for _, p in records)
    best_joint = math.exp(best)
    assert np.allclose(PI, [.25, .125, .625])
    assert np.allclose(A.sum(axis=1), 1) and np.allclose(B.sum(axis=1), 1)
    assert np.allclose(gamma.sum(axis=1), 1) and np.allclose(xi.sum(axis=(1,2)), 1)
    assert np.isclose(total, math.exp(ll)) and np.isclose(max_enum, best_joint)
    assert best_joint <= total + 1e-15 and np.allclose(np.exp(alpha[-1]).sum(), total)
    assert np.allclose(np.exp(alpha + beta - ll), gamma)
    assert np.all(np.isfinite(filt))
    enum_gamma = np.zeros_like(gamma)
    for candidate, value in records:
        for t, state in enumerate(candidate):
            enum_gamma[t, state] += value / total
    assert np.allclose(enum_gamma, gamma)
    assert np.allclose(filt[-1], gamma[-1])
    assert np.allclose(xi.sum(axis=2), gamma[:-1])
    assert np.allclose(xi.sum(axis=1), gamma[1:])
    print("STATE ORDER:", ", ".join(STATES))
    print("OBSERVATIONS:", ", ".join(SYMBOLS[i] for i in OBS))
    print("First-observation prior (Day 0 Cloudy):", PI)
    print("A row sums:", np.array2string(A.sum(1), precision=3))
    print("B row sums:", np.array2string(B.sum(1), precision=3))
    print("Viterbi log-score table (rows t=1..3; columns Sunny,Cloudy,Rainy):")
    print(np.array2string(delta, precision=6, suppress_small=True))
    print("Backpointers (rows t=1..3; -1 denotes initialization):")
    print(back)
    print("Forward log-alpha table:")
    print(np.array2string(alpha, precision=6, suppress_small=True))
    print("Enumerated paths:", len(records))
    print(f"Best path: {' → '.join(STATES[i] for i in path)}")
    print(f"Best path joint probability: {best_joint:.9f}")
    print(f"Observation evidence P(O): {total:.9f}")
    print(f"Best-path posterior P(Q*|O): {best_joint/total:.9f}")
    print("Smoothed gamma P(q_t|O), rows t=1..3:")
    print(np.array2string(gamma, precision=6, suppress_small=True))
    print("Filtering posterior P(q_t|o_1:t), rows t=1..3:")
    print(np.array2string(filt, precision=6, suppress_small=True))
    print("xi transition posterior sums:", np.array2string(xi.sum((1,2)), precision=6))
    print("Check: P(O|Q*) =", f"{np.prod([B[path[t], OBS[t]] for t in range(len(OBS))]):.9f}")
    print("Check: P(Q*,O) = P(Q*)P(O|Q*) =", f"{best_joint:.9f}")
    print("Check: P(Q*|O) = P(Q*,O)/P(O) =", f"{best_joint/total:.9f}")
    marginal_path = np.argmax(gamma, axis=1)
    marginal_joint = PI[marginal_path[0]] * B[marginal_path[0], OBS[0]]
    for t in range(1, len(OBS)):
        marginal_joint *= A[marginal_path[t-1], marginal_path[t]] * B[marginal_path[t], OBS[t]]
    print("Marginal-MAP states:", " → ".join(STATES[i] for i in marginal_path))
    print(f"Marginal-MAP sequence joint score: {marginal_joint:.9f}")
    # Source-provided illustrative speech-HMM values, kept distinct from the GMM experiment.
    p_emissions = .76*.27*.84
    p_joint = .7*.4*p_emissions
    print("Illustrative hi example (provided values): emission product =", f"{p_emissions:.6f}")
    print("Illustrative hi example: transition-emission product (a12=.7,a22=.4) =", f"{p_joint:.8f}")
    print("Illustrative hi example: with explicit exit factor .6 =", f"{p_joint*.6:.9f}")
    w,m,v,x,g,p,d,bk,bestg,gg,xx,ff,llg = hmm_gmm_demo()
    print("HMM-GMM normalized log-energy features:", np.array2string(x, precision=2))
    print("HMM-GMM Viterbi state path:", " → ".join(SPEECH_STATES[i] for i in p))
    print(f"HMM-GMM log evidence: {llg:.6f}")
    print("Speech topology: fixed 6-frame horizon; final state absorbing; no exit factor.")
    # Independent two-observation practice example (third symbol completes row normalization).
    pa = np.array([[.7, .3], [.2, .8]])
    pb = np.array([[.5, .2, .3], [.1, .9, 0.]])
    _, _, _, _, _, practice_ll = forward_backward(np.array([.6, .4]), pa, pb, (0,1))
    practice = enumerate_paths(np.array([.6,.4]), pa, pb, (0,1))
    practice_total = math.exp(practice_ll)
    assert np.isclose(sum(v for _,v in practice), practice_total)
    print(f"Two-step practice: evidence={practice_total:.6f}; best joint={max(v for _,v in practice):.6f}; best posterior={.081/practice_total:.6f}")

if __name__ == '__main__':
    run()
