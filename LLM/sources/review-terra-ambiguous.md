# Audit of ambiguous LLM exam answers

Independent review of all 62 records currently marked ambiguous. The JSON file contains item-level derivations, comparisons with the existing notes, proposed answers and status. The answer choices shown in JSON for convention-dependent items identify the course or examination interpretation, not an assertion that alternatives are impossible. For questions that prescribe −1 when information is insufficient, that prescribed response is derivable. `current_note_accurate` assesses the factual caveat itself; a true value does not mean the ambiguous status is warranted.

| Week | Resolvable | Defective | Convention | Total |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 7 | 6 | 4 | 17 |
| 2 | 4 | 5 | 5 | 14 |
| 3 | 7 | 1 | 6 | 14 |
| 4 | 12 | 4 | 1 | 17 |
| **Total** | **30** | **16** | **16** | **62** |

Recommended status changes to derived: **30**. Convention-dependent entries remain ambiguous when the prompt does not settle the convention, even where the course or key suggests a preferred answer.

Material answer changes: `da5004_2024T1_Q1_NA--q192` (C is defensible; B is unproven), `da5004_2025T3_ET_AN--q17` and `da5004_2025T3_ET_FN--q25` (B and C under the course attention-product convention), `da5004_2025T3_ET_FN--q8` (B constructs the attention matrix; C consumes it), `da5004_2026T1_ET_FN--q16` and `da5004_2026T2_ET_FN--q16` (C only: B is not necessary), `da5004_2026T2_ET_AN--q18` (C), and `da5004_2026T2_ET_AN--q19` (B). The last two original PNGs show 3 and 6, respectively, where the inventory transcribes 13 and 16.

The corpus-based results for `da5004_2025T1_Q1_NA--q119` and `--q120` are 100/41 ≈ 2.4390, narrowly beyond the printed answer-key upper bound 2.43; this is a key-tolerance discrepancy, not an ambiguity in the calculation. The pictured matrices in `da5004_2025T2_Q1_NA--q147`–`--q150` also differ from their Markdown transcriptions.
