---
title: Learning-Rate Warmup
tags: [llm/concept, transformer, optimization]
---

# Learning-Rate Warmup

**Formal definition.** Learning-rate warmup increases the optimizer's step-size multiplier during an initial training interval. Vaswani et al. (2017) use the step-dependent schedule

$$\eta(s)=d_{\mathrm{model}}^{-1/2}\min\left(s^{-1/2},\;s\,w^{-3/2}\right),\qquad s\ge1,$$

where $s$ is the optimizer step and $w$ is the warmup duration. The rate rises linearly through step $w$, where the two terms coincide, and then decays proportionally to $s^{-1/2}$. The original Transformer paper specifies $w=4000$; both the schedule and this value come from that paper. Here $d_{\mathrm{model}}$ is the representation width, and $s,w$ are dimensionless step counts.

**Why it matters here.** Warmup moderates early updates while the model and optimizer statistics develop, and the subsequent decay reduces the nominal learning rate over training. Adam's adaptive parameter-wise scaling means $\eta(s)$ alone is not the magnitude of every parameter update. The ebook's categorical claim that post-normalized training is unstable without warmup is too broad: stability depends on architecture, initialization, optimizer and task. Warmup is a schedule choice, not a proof of convergence or a universal requirement.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[layer-normalization]]
- [[residual-connections]]
- [[label-smoothing]]
