---
title: "Speech Technology: Exam Study Route for Weeks 1–4"
date: 2026-11-10
status: complete
tags: [speech/moc, speech/exam-study]
map: "[[Knowledge-Map]]"
---

# Exam Study Route — Weeks 1–4

[[Knowledge-Map|Knowledge Map]] · [[exam-question-index|Paper inventory and source audit]]

These guides teach the confirmed Week 1–4 curriculum from the beginning and allocate additional explanation and practice to mechanisms tested in the supplied papers. They complement the full weekly references and authentic question banks. Week 5 is outside this evidence-based series because the current question-routing task covered only Weeks 1–4.

## Choose a guide

| Week | Learn from the beginning | Main additional practice | Authentic bank |
| :--- | :--- | :--- | :--- |
| 1 | [[week-1-exam-focused-notes\|Signals, sampling, systems and speech production]] | Sampling/unit conversions; proofs of linearity and time invariance | [[week-1-exam-questions\|18 question occurrences]] |
| 2 | [[week-2-exam-focused-notes\|Perception, short-time analysis, cepstrum, MFCCs and Gaussian review]] | Explain MFCC stages; window trade-offs; Gaussian density facts | [[week-2-exam-questions\|21 question occurrences]] |
| 3 | [[week-3-exam-focused-notes\|Gaussian decisions, GMMs, EM, clustering and Markov chains]] | GMM calculations; K-means/LBG iterations; weather-path traceback | [[week-3-exam-questions\|38 question occurrences]] |
| 4 | [[week-4-exam-focused-notes\|HMMs, Viterbi, Forward and speech acoustic models]] | Hidden-state score tables; max versus sum; model roles and phone-state counts | [[week-4-exam-questions\|21 question occurrences]] |

## What the priorities mean

- **P1 — extra study and repeated application:** substantial or repeated direct testing, or an essential distinction needed to solve those questions. Learn the mechanism, explain the reason, and practise changed numbers.
- **P2 — complete study and concise rehearsal:** direct testing mainly through short conceptual prompts or supporting distinctions. Be able to give an accurate explanation, not just an answer letter.
- **P3 — complete syllabus coverage, less direct paper evidence:** a named curriculum topic or necessary supporting mechanism with no direct standalone item located in this corpus. Learn it; allocate fewer repetitions once you can apply it correctly.

**Dependency takes precedence over priority.** A P3 source–filter explanation may need to be read before a P1 MFCC pipeline. Priority is a revision-effort label, not an instruction to skip sections or a claim that one concept is inherently unimportant.

## How the evidence was measured

All 200 academic records across the 18 supplied PDF files were reviewed. Ninety-eight occurrences contribute to Weeks 1–4; 102 are outside the current scope. Two PDFs contain administrative notices instead of academic papers. An occurrence is one source question record: repeated wording and FN/AN variants are preserved as provenance, so occurrence counts do not equal unique problems.

The tables also report **distinct exam–terms**, grouping by `(year/term, Q1 or Q2 or ET)` and merging FN/AN variants in that group. Repeated templates can still occur in different terms; that does not make them independent samples from a future exam. The tables assign every included occurrence one primary topic family so each week's counts reconcile with its bank. Questions often touch other concepts too; the classification is for evidence accounting, not a claim that concepts are isolated.

**Do not turn the counts into predicted marks.** The corpus is small and uneven, the curriculum ordering can change, and the same question can combine several ideas. One mixed Transformer-ASR item contributes only its explicitly requested STFT front end; its full marks are not treated as Week 2 marks. Priorities therefore use both observed task types and prerequisite value. Detailed source IDs and counts are available in [the study-priority manifest](sources/exam-papers/exam-study-priorities.json).

The learner's exam-scope rule remains: **Q1 Weeks 1–4; Q2 Weeks 1–8 with emphasis on 5–8; ET all 12 weeks with emphasis on 9–12.** A Week 1–4 question found in ET still informs the corresponding topic's preparation. The four guides together support Q1 preparation; they are not sufficient on their own for all Q2 or ET content.

## A first-time learner's study method

1. **Read in dependency order.** Start at Week 1 and follow each guide's learning order. Define unfamiliar symbols before substituting numbers. The full weekly notes provide additional derivations and figures when needed.
2. **Work before checking.** At each example, cover its answer, state the appropriate rule, and attempt the next line. If you cannot begin, return to the definition; copying a solution does not establish recall.
3. **Explain the mechanism aloud.** Answer “why this operation?” and “what do these numbers mean?” For example, an MFCC explanation should justify the log and DCT, not merely list them.
4. **Use authentic questions after the first explanation.** Begin with one representative question per family, follow its source link, and reproduce the solution without viewing the bank's answer. Then try a changed-number example.
5. **Revisit errors across sessions.** Record the error as units, concept, formula choice, arithmetic, indexing, missing step or source ambiguity. Repeat the same mechanism with new values after a break and again in the next study session.
6. **Finish the P3 coverage.** Use the guides' checks to ensure an absence of located questions has not left a gap in the confirmed curriculum.
7. **Run a mixed closed-book pass.** Choose questions from different families so you must identify the method before computing. Do not practise only a long block of problems that all announce the same algorithm.

## What “ready” means

| Self-assessment | Evidence | Next action |
| :--- | :--- | :--- |
| Recognition only | The answer looks familiar when shown | Read the explanation and attempt the first step unaided |
| Reproduction | Can repeat a worked solution with the same numbers | Change a number, initial condition or requested quantity |
| Transfer | Can solve the changed problem and justify the method | Mix with another topic and check units/assumptions |
| Explanation | Can teach the mechanism and name its failure cases | Maintain with shorter retrieval practice |

Treat a P1 topic as ready only after transfer and explanation. For P2/P3, ensure at least accurate unaided explanation and the basic calculations shown in the guide; remedy any prerequisite weakness before moving forward. Familiarity with a previous answer letter is not readiness.

## Important corrections to keep while studying

- Global non-stationarity and local quasi-stationarity are compatible; the supplied STFT MCQ does not cleanly distinguish them.
- The conventional Gaussian parameters are mean and variance, but the supplied alternatives with median/mode and scalar covariance are not mathematically unique.
- A Gaussian density is not probability at an exact point; GMM responsibility is not an unweighted component density.
- Soft EM is different from assigning every point to its single highest-scoring component.
- Viterbi returns a best path; Forward sums path contributions; Forward–Backward gives smoothed occupancies used for soft training.
- The 2026 two-state HMM has two tied best paths; the older weather variants have different known starts and observations.
- HMM–GMM emission roles, tri-state temporal stages and triphone context are separate distinctions.

All authentic-bank solutions are independently derived; the PDFs do not supply verifiable academic answer keys. Defective options or missing diagram parameters remain explicitly marked. The study guides teach the valid underlying rule instead of asserting certainty about an unavailable official key.

## Reproducibility

Existing weekly figures are reused with captions. No new experimental speech data or recognition-accuracy claims are introduced. Each guide includes one optional, fully executed computational check; beginners can study the hand calculations without programming.

From the Speech Technology track:

```sh
uv run python build_exam_study_priorities.py
uv run python verify_exam_focused_notes.py
```

The verifier checks evidence reconciliation, Markdown/YAML/navigation, figure references and captions, executed code outputs, and independent numerical examples. [Verification output](sources/exam-papers/exam-focused-verification.txt).
