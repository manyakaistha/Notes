# Routing review — Weeks 1–4

## Scope and counts

Compared all 488 routing records with the 488 question-inventory records, `topics.json`, and the Week 1–4 ebook transcripts. Routing totals reconcile: W1 73, W2 65, W3 36, W4 67, archive 247. Suspected in-scope routing errors: **3 records** (**2 high**, **1 medium** confidence). Exact-stem duplicate groups with topic differences: **2** (listed below). No archive record lacked `likely_later_week` (including `other/unknown`).

## Suspected misroutes

| ID | Current routing | Suggested routing | Evidence (quote from stem) |
|---|---|---|---|
| `da5004_2025T3_ET_AN--q17`; `da5004_2025T3_ET_FN--q25` | W1 · `w1-self-attention-qkv` | W1 · `w1-scaled-dot-product` + `w1-ffn-encoder-stack` | “sequence length is doubled from $L$ to $2L$” and asks self-attention and FFN FLOP factors. The distinguishing ideas are quadratic attention scaling and linear-in-sequence FFN scaling; current Q/K/V-projection topic omits both tested cost components. **High.** |
| `da5004_2024T3_ET_AN--q184` | W1 · `w1-self-attention-qkv` | W2 · `w2-positional-encoding` (with W1 Q/K projection as a secondary topic if multi-topic routing is intended) | “correct expression for $q_{ij}$” with answer choices expanding terms containing both $x_i,x_j$ and positional vectors $p_i,p_j$. This is an input-plus-position embedding expansion, aligning with Week 2 positional encoding; present week/topic misses that explicit content. **Medium** because the positional scheme is unspecified and the algebra also uses Week 1 Q/K projections. |

## Archived-record audit

Archive disposition totals by `likely_later_week`: W5 **71**, W7 **42**, W8 **36**, W11 **46**, W12 **44**, other/unknown **8**. There are **98** distinct reason strings; the largest reason families are beyond-sinusoidal positional methods (**22**), efficient/sparse attention or inference caching (**20**), BART/T5/prefix-LM objectives and comparisons (**13**), WordPiece (**10**), and tokenizer/subword methods (**10**). The remaining strings are a long tail of similarly specific tokenizer, later architecture, data/scaling, position, and efficient-attention reasons. No text-verifiable archive record showed a supported reason mismatch or a question wholly testing a confirmed W1–4 topic. In particular, the BERT preprocessing records mix Week 4 special-token/position inputs with Week 5 subword-tokenizer content, so archiving them is defensible under the within-week rule.

**Not text-verifiable (image-only archived records; 68):**

`da5004_2024T1_ET_AN--q147`, `da5004_2024T1_ET_FN--q204`, `da5004_2024T1_ET_FN--q205`, `da5004_2024T1_Q1_NA--q188`, `da5004_2024T1_Q2_NA--q134`, `da5004_2024T3_ET_AN--q185`, `da5004_2024T3_ET_AN--q187`, `da5004_2024T3_ET_AN--q188`, `da5004_2024T3_ET_AN--q194`, `da5004_2024T3_ET_AN--q195`, `da5004_2024T3_ET_FN--q202`, `da5004_2024T3_ET_FN--q204`, `da5004_2024T3_ET_FN--q205`, `da5004_2024T3_ET_FN--q207`, `da5004_2024T3_ET_FN--q208`, `da5004_2024T3_ET_FN--q209`, `da5004_2024T3_Q2_NA--q140`, `da5004_2024T3_Q2_NA--q142`, `da5004_2024T3_Q2_NA--q143`, `da5004_2024T3_Q2_NA--q144`, `da5004_2024T3_Q2_NA--q145`, `da5004_2024T3_Q2_NA--q146`, `da5004_2024T3_Q2_NA--q147`, `da5004_2024T3_Q2_NA--q151`, `da5004_2025T1_Q2_NA--q128`, `da5004_2025T1_Q2_NA--q129`, `da5004_2025T1_Q2_NA--q130`, `da5004_2025T1_Q2_NA--q132`, `da5004_2025T1_Q2_NA--q133`, `da5004_2025T1_Q2_NA--q134`, `da5004_2025T1_Q2_NA--q135`, `da5004_2025T2_ET_AN--q202`, `da5004_2025T2_ET_AN--q203`, `da5004_2025T2_ET_AN--q206`, `da5004_2025T2_ET_AN--q207`, `da5004_2025T2_ET_AN--q208`, `da5004_2025T2_ET_FN--q149`, `da5004_2025T2_ET_FN--q150`, `da5004_2025T2_ET_FN--q153`, `da5004_2025T2_ET_FN--q154`, `da5004_2025T2_Q2_NA--q279`, `da5004_2025T2_Q2_NA--q280`, `da5004_2025T2_Q2_NA--q281`, `da5004_2025T2_Q2_NA--q283`, `da5004_2025T2_Q2_NA--q284`, `da5004_2025T2_Q2_NA--q285`, `da5004_2025T3_ET_AN--q18`, `da5004_2025T3_ET_AN--q21`, `da5004_2025T3_ET_AN--q22`, `da5004_2025T3_ET_FN--q19`, `da5004_2025T3_ET_FN--q20`, `da5004_2025T3_ET_FN--q21`, `da5004_2025T3_ET_FN--q22`, `da5004_2025T3_Q1_NA--q192`, `da5004_2025T3_Q2_NA--q107`, `da5004_2025T3_Q2_NA--q108`, `da5004_2025T3_Q2_NA--q111`, `da5004_2025T3_Q2_NA--q112`, `da5004_2026T1_ET_AN--q9`, `da5004_2026T1_ET_AN--q12`, `da5004_2026T1_ET_FN--q5`, `da5004_2026T1_Q2_NA--q16`, `da5004_2026T2_ET_AN--q7`, `da5004_2026T2_ET_AN--q8`, `da5004_2026T2_ET_AN--q16`, `da5004_2026T2_ET_AN--q20`, `da5004_2026T2_ET_AN--q21`, `da5004_2026T2_ET_FN--q5`

## Duplicate-stem consistency

Two normalized exact-stem groups have different topic IDs:

1. “Choose the correct statements” — `da5004_2024T1_Q1_NA--q194` (W3: `w3-finetuning`, `w3-pretraining-objective`) and `da5004_2025T1_Q1_NA--q108` (W3: `w3-finetuning`, `w3-gpt1-architecture`). The stem is generic; these are not substantive duplicates, and the options test distinct details.
2. “Choose the correct statements regarding an encoder-decoder transformer model:” — `da5004_2024T3_ET_AN--q191` (W2: `w2-full-transformer`, `w2-masked-self-attention`) and `da5004_2024T3_ET_FN--q190` (W2: `w2-full-transformer`, `w2-teacher-forcing`). Their options differ and each topic set matches the statements asked. Both groups stay in the same week; neither is a week-level inconsistency.

## Confidence flags

Routing marks **26 medium** and **1 low** confidence records. These flags were retained; except for the q184 issue above, they did not independently establish a routing error:

`da5004_2024T1_ET_AN--q149` (medium), `da5004_2024T1_ET_FN--q202` (medium), `da5004_2024T1_Q1_NA--q193` (medium), `da5004_2024T1_Q2_NA--q143` (medium), `da5004_2024T3_ET_AN--q184` (medium), `da5004_2024T3_Q1_NA--q279` (medium), `da5004_2024T3_Q1_NA--q280` (medium), `da5004_2024T3_Q1_NA--q289` (medium), `da5004_2024T3_Q2_NA--q151` (medium), `da5004_2024T3_Q2_NA--q154` (medium), `da5004_2025T1_Q1_NA--q111` (medium), `da5004_2025T1_Q2_NA--q129` (medium), `da5004_2025T2_ET_AN--q195` (medium), `da5004_2025T2_ET_FN--q142` (medium), `da5004_2025T2_Q1_NA--q135` (medium), `da5004_2025T2_Q1_NA--q141` (medium), `da5004_2025T2_Q2_NA--q279` (medium), `da5004_2025T3_ET_AN--q11` (medium), `da5004_2025T3_ET_FN--q2` (medium), `da5004_2025T3_ET_FN--q8` (medium), `da5004_2025T3_ET_FN--q13` (medium), `da5004_2025T3_Q1_NA--q192` (low), `da5004_2025T3_Q2_NA--q122` (medium), `da5004_2026T1_ET_AN--q6` (medium), `da5004_2026T1_Q1_NA--q12` (medium), `da5004_2026T2_ET_AN--q17` (medium), `da5004_2026T2_ET_FN--q13` (medium)

The sole low-confidence record is `da5004_2025T3_Q1_NA--q192`; its inventory stem is truncated and is not text-verifiable as a question. Its archive reason says the source stem/options are absent, consistent with the available record.
