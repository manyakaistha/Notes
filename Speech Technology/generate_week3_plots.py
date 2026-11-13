"""Generate reproducible Week 3 Gaussian, soft-EM, and weather-chain figures."""
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from week3_lab import gaussian_em, normal_pdf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "week 3" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 180, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def main():
    # 1: equal-prior binary Gaussian classifier; shaded class-conditional errors.
    x = np.linspace(-4, 6, 1000)
    p0, p1 = normal_pdf(x, 0, 1), normal_pdf(x, 2, 1)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x, p0, label=r"$p(x|C_0)=\mathcal{N}(0,1)$")
    ax.plot(x, p1, label=r"$p(x|C_1)=\mathcal{N}(2,1)$")
    ax.fill_between(x, 0, .5*p0, where=x >= 1, alpha=.28, label=r"$C_0$ error contribution")
    ax.fill_between(x, 0, .5*p1, where=x < 1, alpha=.28, label=r"$C_1$ error contribution")
    ax.axvline(1, color="black", ls="--", label="Bayes threshold $x=1$")
    ax.set(xlabel="Scalar feature $x$ (standardized units)", ylabel="Probability density (unit$^{-1}$)", title="Equal-prior Gaussian classification and Bayes error")
    ax.legend(fontsize=10, ncol=2)
    save(fig, "01_binary_gaussian_bayes_error.png")

    # 2: correlated bivariate Gaussian, one-standard-deviation Mahalanobis contour.
    mean = np.array([0.5, -0.25]); cov = np.array([[1., .65], [.65, 1.5]])
    u = np.linspace(-3.5, 4.5, 240); v = np.linspace(-4.0, 3.5, 240)
    xx, yy = np.meshgrid(u, v); pts = np.stack([xx-mean[0], yy-mean[1]], axis=-1)
    inv = np.linalg.inv(cov); det = np.linalg.det(cov)
    q = np.einsum("...i,ij,...j->...", pts, inv, pts)
    density = np.exp(-.5*q)/(2*np.pi*np.sqrt(det))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cf = ax.contourf(xx, yy, density, levels=18, cmap="mako")
    ax.contour(xx, yy, q, levels=[1., 4., 9.], colors="white", linewidths=1.5)
    ax.scatter(*mean, color="gold", edgecolor="black", zorder=3, label="Mean")
    fig.colorbar(cf, ax=ax, label="Joint density (feature-unit$^{-2}$)")
    ax.set(xlabel="$x_1$ (standardized feature units)", ylabel="$x_2$ (standardized feature units)", title="Bivariate Gaussian: covariance tilts equal-density ellipses")
    ax.legend()
    save(fig, "02_bivariate_gaussian_covariance.png")

    # 3: weighted GMM density and component contributions, fixed parameters.
    grid = np.linspace(-3, 7, 900); weights = np.array([.6, .4]); means = np.array([1., 3.]); vars_ = np.array([1., 1.])
    components = np.column_stack([weights[k]*normal_pdf(grid, means[k], vars_[k]) for k in range(2)])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(grid, components[:, 0], label=r"$w_1 p_1(x)$: $w_1=0.6,\ \mu_1=1,\ \sigma_1=1$")
    ax.plot(grid, components[:, 1], label=r"$w_2 p_2(x)$: $w_2=0.4,\ \mu_2=3,\ \sigma_2=1$")
    ax.plot(grid, components.sum(axis=1), color="black", lw=2.5, label="Mixture density $p(x)$")
    ax.set(xlabel="Scalar feature $x$ (standardized units)", ylabel="Probability density (unit$^{-1}$)", title="A Gaussian mixture is a weighted sum of densities")
    ax.legend(fontsize=10)
    save(fig, "03_gmm_density_decomposition.png")

    # Seeded two-component data and full-batch soft EM.
    rng = np.random.default_rng(42)
    z = rng.choice(2, size=600, p=[.58, .42])
    sample = rng.normal(np.where(z == 0, -1.25, 1.65), np.where(z == 0, .48, .72))
    w, mu, var, ll, resp = gaussian_em(sample)
    order = np.argsort(sample)
    # 4: posterior responsibility traces across ordered observations.
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(np.arange(len(sample)), sample[order], c=resp[order, 1], cmap="viridis", s=17, alpha=.8)
    ax.axhline(mu[0], color="tab:blue", ls="--", label=f"Fitted $\\mu_1={mu[0]:.2f}$")
    ax.axhline(mu[1], color="tab:orange", ls="--", label=f"Fitted $\\mu_2={mu[1]:.2f}$")
    ax.set(xlabel="Observation rank by feature value (index)", ylabel="Observed feature $x$ (standardized units)", title="Soft posterior assignment varies continuously across overlap")
    fig.colorbar(ax.collections[0], ax=ax, label=r"Posterior responsibility $r_{i2}$")
    ax.legend()
    save(fig, "04_soft_em_responsibilities.png")

    # 5: fitted GMM density alongside the actual generated sample histogram.
    grid = np.linspace(-3.2, 4.2, 700)
    fit = sum(w[k]*normal_pdf(grid, mu[k], var[k]) for k in range(2))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.histplot(x=sample, bins=35, stat="density", color="steelblue", alpha=.3, edgecolor=None, ax=ax, label="Seeded sample")
    ax.plot(grid, fit, color="black", lw=2.5, label="Soft-EM fitted mixture")
    for k in range(2):
        ax.plot(grid, w[k]*normal_pdf(grid, mu[k], var[k]), ls="--", label=f"Component {k+1}: $w={w[k]:.2f}$")
    ax.set(xlabel="Scalar feature $x$ (standardized units)", ylabel="Density (unit$^{-1}$)", title="Soft EM estimates a two-component density from observations")
    ax.legend(fontsize=10)
    save(fig, "05_gmm_soft_em_fit.png")

    # 6: observed-data log-likelihood trace, computed from the same EM run.
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(np.arange(len(ll)), ll, marker="o", ms=4)
    ax.set(xlabel="EM iteration (count)", ylabel="Observed log-likelihood $\\ell$ (nats)", title="Full-batch soft EM likelihood across iterations")
    save(fig, "06_soft_em_log_likelihood.png")

    # 7: slide's explicitly assumed forecast matrix; exhaustive 27-path optimum.
    states = ("Sunny", "Cloudy", "Rainy")
    P = np.array([[.5, .375, .125], [.25, .125, .625], [.25, .075, .675]])
    paths = list(product(range(3), repeat=3)); start = 0
    scores = [float(np.prod([P[a, b] for a, b in zip((start,)+p[:-1], p)])) for p in paths]
    best = max(range(len(paths)), key=lambda j: (scores[j], tuple(-i for i in paths[j])))
    greedy=[]; state=start
    for _ in range(3):
        state=int(np.argmax(P[state])); greedy.append(state)
    best_path=paths[best]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for day in range(4):
        for s, name in enumerate(states):
            ax.scatter(day, s, s=500, color=sns.color_palette()[s], edgecolor="white", zorder=3)
            ax.text(day, s, name[0], ha="center", va="center", color="white", weight="bold", zorder=4)
    for d in range(1,4):
        ax.plot([d-1,d],[greedy[d-1] if d>1 else start, greedy[d-1]], color="gray", ls=":", lw=2)
        ax.plot([d-1,d],[best_path[d-2] if d>1 else start, best_path[d-1]], color="crimson", lw=3)
    ax.text(.04, .97, "red: globally best path  |  gray dotted: greedy", transform=ax.transAxes, va="top", fontsize=11)
    ax.set(xticks=range(4), xticklabels=["Day 0\nknown", "Day 1", "Day 2", "Day 3"], yticks=range(3), yticklabels=states, ylabel="Weather state (categorical)", title=f"Three-step weather forecast: exhaustive optimum beats greedy ({scores[best]:.3f} vs {np.prod([P[a,b] for a,b in zip((start,)+tuple(greedy[:-1]),greedy)]):.3f})")
    ax.set_ylim(-.5, 2.5)
    save(fig, "07_weather_greedy_vs_global_path.png")

    # Matrix rendered with explicit conditioning direction: row = yesterday.
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    sns.heatmap(P, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, square=True,
                xticklabels=states, yticklabels=states, cbar_kws={"label": "Transition probability"}, ax=ax)
    ax.set(xlabel="Today (next state)", ylabel="Yesterday (current state)", title="Assumed weather transition matrix $P_{ij}=P(q_t=j|q_{t-1}=i)$")
    save(fig, "08_weather_transition_matrix.png")

    print(f"Generated {len(list(OUT.glob('*.png')))} Week 3 figures in {OUT}")


if __name__ == "__main__":
    main()
