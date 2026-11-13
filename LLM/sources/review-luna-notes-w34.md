# Luna Review — Weeks 3–4 Notes

Scope reviewed: `week-3-notes.md`, `week-3-exam-focused-notes.md`, `week-4-notes.md`, `week-4-exam-focused-notes.md`, and the 13 requested W3–W4 concept notes. Cross-checks included the W3/W4 archived code outputs and `NOTES.md`, transcript page markers, ebook discrepancy claims, and `routing.json` question IDs. Parameter totals were independently recomputed with the specified `cd LLM && uv run python -c ...` workflow using PyTorch modules.

| File:line | Severity | Problem | Suggested fix / evidence |
|---|---|---|---|
| — | — | No evidenced errors, warnings, or nits found in the reviewed scope. | No change suggested. All 13 concept notes and the four requested study-note files are clean on the checked claims. |

## Verification summary

- Counts: **0 errors, 0 warnings, 0 nits**.
- Recomputed parameter totals: GPT‑1 **116,534,784**; BERT-base **109,482,240**; BERT-large **335,141,888**. Each matched the archived output and stated counting convention.
- Checked the ten self-check answers in each exam-focused note. The arithmetic and stated assumptions are correct, including perplexity, shifted causal targets, beam scores, length normalization, nucleus inclusion, temperature, MLM rates, embeddings, and parameter counts.
- The 80/10/10 BERT corruption percentages are correctly stated as conditional on selecting the 15% prediction positions; loss includes all selected positions. NSP construction and the caveat about random negatives are accurate.
- The fine-tuning equation is correctly distinguished from norm regularization: GPT‑1's $L_3=L_2+\lambda L_1$ combines supervised and auxiliary language-model log-likelihoods; the separate executed L1/L2 norm example is explicitly identified as a different toy objective.
- The listed Week 3 and Week 4 ebook corrections were checked against the transcript's `<!-- PDF p.N` markers and their underlying equations/arithmetic. The cited PDF page numbers match the transcript markers. The correction claims reviewed are supported; `ebook-discrepancies.md` continues to label the registry unverified, which is consistent with it being an inventory rather than an adjudication.
- Every question ID linked from the two main notes and two exam-focused notes exists in `routing.json` for the corresponding week (W3: 36/36 main-note IDs and 14/14 exam-note IDs; W4: 67/67 and 21/21).
- Archived numerical outputs referenced for these checks are consistent with the written calculations and conventions.
