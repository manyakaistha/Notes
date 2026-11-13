"""Deterministic Week 3 probability, Gaussian-mixture, and Markov-chain lab."""
from itertools import product
import math
import numpy as np


def normal_pdf(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def gaussian_em(x, iterations=30):
    """Two-component univariate full-batch soft EM with variance floor."""
    weights = np.array([0.5, 0.5])
    means = np.quantile(x, [0.25, 0.75])
    variances = np.full(2, np.var(x))
    history = []
    responsibilities = None
    for _ in range(iterations + 1):
        densities = np.column_stack([
            weights[k] * normal_pdf(x, means[k], variances[k]) for k in range(2)
        ])
        totals = np.maximum(densities.sum(axis=1), np.finfo(float).tiny)
        responsibilities = densities / totals[:, None]
        history.append(float(np.log(totals).sum()))
        if len(history) > iterations:
            break
        nk = responsibilities.sum(axis=0)
        weights = nk / len(x)
        means = (responsibilities * x[:, None]).sum(axis=0) / nk
        variances = (responsibilities * (x[:, None] - means) ** 2).sum(axis=0) / nk
        variances = np.maximum(variances, 1e-6)
    order = np.argsort(means)
    return weights[order], means[order], variances[order], np.asarray(history), responsibilities[:, order]


def main():
    # Equal-prior Bayes decision for N(0,1) and N(2,1): crossing at x=1.
    error = 0.5 * math.erfc(1 / math.sqrt(2))
    print(f"Equal-prior threshold: x=1.000; Bayes error={error:.6f}")

    # Positive-definite bivariate Gaussian; covariance .65 implies rho=.65/sqrt(1.5).
    covariance = np.array([[1.0, 0.65], [0.65, 1.5]])
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues > 0), "Covariance must be positive definite"
    print(f"Bivariate covariance eigenvalues: {np.array2string(eigenvalues, precision=6)}")

    # A two-component GMM density evaluated at a fixed acoustic feature x.
    x0 = 2.0
    terms = np.array([0.6 * normal_pdf(x0, 1.0, 1.0), 0.4 * normal_pdf(x0, 3.0, 1.0)])
    print(f"Fixed GMM at x=2.0: weighted component densities={np.array2string(terms, precision=6)}; mixture={terms.sum():.6f}")
    print(f"Fixed GMM posterior responsibilities: {np.array2string(terms / terms.sum(), precision=6)}")

    # One exact soft-EM update for the symmetric worked example in the note.
    toy = np.array([0., 2., 4.])
    joint = np.column_stack([0.5 * normal_pdf(toy, mean, 1.) for mean in (0., 4.)])
    toy_resp = joint / joint.sum(axis=1, keepdims=True)
    toy_nk = toy_resp.sum(axis=0)
    toy_mu = (toy_resp * toy[:, None]).sum(axis=0) / toy_nk
    toy_var = (toy_resp * (toy[:, None] - toy_mu) ** 2).sum(axis=0) / toy_nk
    assert np.allclose(toy_nk, [1.5, 1.5])
    print(f"One soft-EM update: effective counts={toy_nk}; weights={toy_nk/3}")
    print(f"One soft-EM update: means={np.array2string(toy_mu, precision=6)}; variances={np.array2string(toy_var, precision=6)}")

    # Seeded complete-data sample; soft EM estimates parameters from observations only.
    rng = np.random.default_rng(42)
    z = rng.choice(2, size=600, p=[0.58, 0.42])
    sample = rng.normal(np.where(z == 0, -1.25, 1.65), np.where(z == 0, 0.48, 0.72))
    weights, means, variances, ll, resp = gaussian_em(sample)
    assert np.all(np.diff(ll) >= -1e-10), "Observed-data log-likelihood should not decrease beyond roundoff"
    assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-12), "Posterior responsibilities must normalize"
    assert np.all(weights > 0) and np.isclose(weights.sum(), 1), "Mixture weights must be positive and sum to one"
    print(f"Synthetic sample: N={sample.size}; generator seed=42; true weights=[0.58, 0.42]")
    print(f"Soft-EM estimates (ordered by mean): weights={np.array2string(weights, precision=4)}, means={np.array2string(means, precision=4)}, variances={np.array2string(variances, precision=4)}")
    print(f"Observed log-likelihood: start={ll[0]:.6f}; end={ll[-1]:.6f}; minimum increment={np.diff(ll).min():.3e}")
    print(f"Responsibility checks: row-sum max error={np.max(np.abs(resp.sum(axis=1)-1)):.2e}; effective counts={np.array2string(resp.sum(axis=0), precision=2)}")
    hard = np.bincount(resp.argmax(axis=1), minlength=2) / sample.size
    print(f"Hard-assignment fractions from final posterior: {np.array2string(hard, precision=4)}")

    # Slide's illustrative transition matrix is conditioned on yesterday's state.
    states = ("Sunny", "Cloudy", "Rainy")
    transition = np.array([[0.5, 0.375, 0.125], [0.25, 0.125, 0.625], [0.25, 0.075, 0.675]])
    observed = ("Sunny", "Cloudy", "Rainy", "Rainy", "Cloudy", "Sunny", "Sunny", "Rainy", "Sunny", "Cloudy", "Cloudy", "Rainy", "Sunny")
    index = {s: i for i, s in enumerate(states)}
    counts = np.zeros((3, 3), dtype=int)
    for a, b in zip(observed[:-1], observed[1:]):
        counts[index[a], index[b]] += 1
    empirical = counts / counts.sum(axis=1, keepdims=True)
    given = index["Cloudy"]
    path = ("Sunny", "Sunny", "Cloudy", "Rainy", "Cloudy")
    path_prob = 1.0
    prev = "Cloudy"
    for nxt in path:
        path_prob *= transition[index[prev], index[nxt]]
        prev = nxt
    print(f"Observed weather transitions (rows S,C,R): {counts.tolist()}")
    print(f"Count-estimated transition matrix (rows S,C,R): {np.array2string(empirical, precision=3)}")
    print(f"Assumed forecast transition matrix (rows S,C,R): {np.array2string(transition, precision=3)}")
    print(f"P(Sunny|Cloudy): count estimate={empirical[1,0]:.3f}; assumed forecast={transition[1,0]:.3f}; full matrices are distinct")
    print(f"P(next five states S,S,C,R,C | yesterday C)={path_prob:.8f}")
    forecast = np.eye(3)[index["Cloudy"]] @ np.linalg.matrix_power(transition, 3)
    print(f"Three-day marginal from Cloudy (S,C,R): {np.array2string(forecast, precision=6)}")
    sunny_forecast = np.eye(3)[index["Sunny"]] @ np.linalg.matrix_power(transition, 3)
    print(f"Three-day marginal from Sunny (S,C,R): {np.array2string(sunny_forecast, precision=6)}")

    start = index["Sunny"]
    paths = list(product(range(3), repeat=3))
    scored = [(float(np.prod([transition[a, b] for a, b in zip((start,) + p[:-1], p)])), p) for p in paths]
    best_prob, best_path = max(scored, key=lambda item: (item[0], tuple(-i for i in item[1])))
    greedy = []
    state = start
    for _ in range(3):
        state = int(np.argmax(transition[state]))
        greedy.append(state)
    greedy_prob = float(np.prod([transition[a, b] for a, b in zip((start,) + tuple(greedy[:-1]), greedy)]))
    print(f"Sunny-start greedy path: {' -> '.join(states[i] for i in greedy)}; probability={greedy_prob:.6f}")
    print(f"Sunny-start exhaustive best of {len(paths)} paths: {' -> '.join(states[i] for i in best_path)}; probability={best_prob:.6f}")
    # Max-product dynamic programming, with argmax backpointers.
    delta = np.eye(3)[start]
    back = []
    for _ in range(3):
        candidates = delta[:, None] * transition
        back.append(np.argmax(candidates, axis=0))
        delta = candidates.max(axis=0)
    final = int(np.argmax(delta))
    decoded = [final]
    for pointers in reversed(back):
        decoded.append(int(pointers[decoded[-1]]))
    decoded = tuple(reversed(decoded))[1:]
    print(f"Max-product DP path: {' -> '.join(states[i] for i in decoded)}; probability={float(delta.max()):.6f}; matches enumeration={decoded == best_path}")
    assert decoded == best_path and np.isclose(float(delta.max()), best_prob), "Max-product DP must match exhaustive path search"


if __name__ == "__main__":
    main()
