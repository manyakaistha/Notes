---
title: Viterbi Decoding Algorithm
tags: [speech/concept, dynamic-programming, viterbi, decoding]
---

# Viterbi Decoding Algorithm

The **Viterbi Algorithm** is a dynamic programming algorithm that computes the single most probable sequence of hidden states $Q^* = (q_1^*, \dots, q_T^*)$ given an observation sequence $O = (o_1, \dots, o_T)$ and an HMM $\lambda$:

$$Q^* = \arg\max_Q P(Q, O \mid \lambda)$$

### Trellis Recursion
1. **Initialization ($t = 1$):**
   $$\delta_1(j) = \pi_j b_j(o_1), \quad \psi_1(j) = 0$$
2. **Induction ($t = 2 \dots T$):**
   $$\delta_t(j) = \max_{1 \le i \le N} \left[ \delta_{t-1}(i) a_{ij} \right] b_j(o_t)$$
   $$\psi_t(j) = \arg\max_{1 \le i \le N} \left[ \delta_{t-1}(i) a_{ij} \right]$$
3. **Termination ($t = T$):**
   $$P^* = \max_{1 \le i \le N} \delta_T(i), \quad q_T^* = \arg\max_{1 \le i \le N} \delta_T(i)$$
4. **Backtracking:**
   $$q_t^* = \psi_{t+1}(q_{t+1}^*), \quad t = T-1, \dots, 1$$

### Log-Domain Formulation
To prevent arithmetic underflow from multiplying hundreds of probabilities, the algorithm operates in the logarithmic domain:

$$\ln \delta_t(j) = \max_i \left[ \ln \delta_{t-1}(i) + \ln a_{ij} \right] + \ln b_j(o_t)$$

**Appears in:** [[week 2/week-2-notes|Week 2 Notes]]

## Related
- [[hidden-markov-models]]
- [[gaussian-mixture-models]]
