---
title: "HMM State-Transition Topologies"
tags: [speech/concept, hmm, topology]
---

# HMM State-Transition Topologies

**Formal definition.** An HMM topology specifies which state transitions are permitted and therefore which state paths have nonzero probability. An ergodic or fully connected topology permits transitions among states (self-transitions may also be allowed); a left-to-right topology constrains progression toward later states and typically disallows backward jumps. These are model-structure choices that encode assumptions about sequence organization, not different emission distributions. In speech recognition, a phone may be represented by a small chain of emitting states with self-loops, enabling variable duration while preserving an ordered progression. Terminology requires care: a lecture may colloquially define “ergodic” by saying every transition probability is positive, but formal finite Markov-chain ergodicity does not require every one-step matrix entry to be positive; irreducibility and aperiodicity are the relevant chain properties. A left-to-right topology is a common ASR convention, not a universal physical law of speech.

**Why it matters here.** Topology determines legal alignments and constrains Viterbi search; confusing it with acoustic emission modeling leads to incorrect claims about what the HMM can generate.

**Appears in:** [[Knowledge-Map]], [[week 5/week-5-notes|Week 5]]

## Related
- [[hidden-markov-models]]
- [[discrete-continuous-hmm]]
- [[viterbi-algorithm]]
