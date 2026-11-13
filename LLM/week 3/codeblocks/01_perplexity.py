import numpy as np
probs = np.array([0.8, 0.5, 0.25])
ce = -np.log(probs).mean()
ppl_ce = np.exp(ce)
ppl_probs = np.exp(-np.log(probs).mean())
print(f"Mean cross-entropy: {ce:.6f} nats/token")
print(f"PPL from CE: {ppl_ce:.6f}")
print(f"PPL from per-token probabilities: {ppl_probs:.6f}")
print(f"exp(mean NLL) identity: {np.isclose(ppl_ce, ppl_probs)}")
