---
title: Latin Hypercube Sampling (LHS)
slug: latin-hypercube-sampling
tags: [pinn/concept, pinn/sampling, pinn/quasi-monte-carlo]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Latin Hypercube Sampling (LHS)

Latin Hypercube Sampling (LHS) is a stratified statistical quasi-random sampling strategy that guarantees comprehensive multi-dimensional coverage across the spatiotemporal domain $\Omega \times [0, T]$ while avoiding the catastrophic clustering and void formation characteristic of pure pseudo-random Monte Carlo sampling. In an $m$-dimensional domain partitioned into $N$ equal probability intervals along each dimension, LHS enforces that exactly one sample is allocated to each hyper-row and hyper-column through random permutation arrays. When distributing interior collocation points for PINN training, LHS reduces the spatial variance of numerical integration over the residual norm $\|\mathcal{R}(u_\theta)\|_{L_2(\Omega)}^2$ from the standard Monte Carlo rate $\mathcal{O}(N^{-1/2})$ toward the quasi-Monte Carlo rate approaching $\mathcal{O}(N^{-1})$, preventing local under-resolution of physics without locking evaluation points into rigid Cartesian grids.

**Why it matters here:** LHS is the gold-standard spatiotemporal discretization method in mesh-free PINN workflows, ensuring balanced physical constraint enforcement across complex domains.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d3|W3D3]] · [[w4d2|W4D2]]

## Related
- [[collocation-points]]
- [[discretization-error]]
- [[physics-residual]]
