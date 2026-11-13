# Week 4 executable artifacts

## Code blocks

| File | What it shows | Topic ID |
|---|---|---|
| `01_decoding.py` / `.txt` | Greedy and general-width beam search with optional mean-log-probability length normalisation on a toy conditional table. | `w4-exhaustive-greedy`, `w4-beam-search` |
| `02_sampling.py` / `.txt` | NumPy top-k, top-p (include the first token crossing the cumulative threshold), temperature distributions, entropy, and seeded sampling. | `w4-topk-temperature`, `w4-topp-nucleus` |
| `03_mlm.py` / `.txt` | An executable seeded MLM corruption function with 15% selection, 80/10/10 replacement categories, and empirical rates. | `w4-bert-mlm` |
| `04_bert_parameters.py` / `.txt` | Formula and PyTorch module counts for BERT-base/large using 30,522 configured vocabulary entries, alongside the paper's 30,000 vocabulary arithmetic. | `w4-bert-architecture-params` |
| `05_nsp_layout.py` / `.txt` | Constructs consecutive and random-next sentence pairs at 50/50 and displays `[CLS]`/`[SEP]` tokens with segment IDs. | `w4-bert-nsp-inputs` |

## Figures

| File | What it shows | Factual caption suggestion |
|---|---|---|
| `assets/01_greedy_beam_tree.png` | Two-step conditional search tree showing the best path under greedy and beam width 2. | Greedy selects A then b with probability 0.3060, while beam width 2 preserves B and finds B then a with probability 0.4851. |
| `assets/02_topk_topp.png` | Original, top-k, and top-p retained probability mass before renormalization. | For probabilities (0.42, 0.25, 0.15, 0.10, 0.08), top-k 3 and top-p 0.80 both retain the first three outcomes, totaling 0.82 before renormalization. |
| `assets/03_temperature_entropy.png` | Softmax distribution over fixed logits at three temperatures. | For logits (2, 1, 0, −1), entropy increases from 0.455 nats at T=0.5 to 0.948 at T=1 and 1.245 nats at T=2. |
| `assets/04_length_normalisation.png` | Raw sequence log probability versus average log probability. | Raw scores rank the one-token candidate at −1.0 above the three-token candidate at −1.5, whereas mean log scores rank the longer candidate at −0.5 above −1.0. |
| `assets/05_bert_masking_statistics.png` | Seeded empirical MLM selection/replacement frequencies over one million positions. | In one million seeded trials, 14.989% of positions were selected and the selected-token split was approximately 79.93% MASK, 10.13% random, and 9.94% unchanged. |
| `assets/06_bert_parameter_breakdown.png` | Embedding, encoder, and pooler parameter counts for BERT-base and large. | The configured BERT-base count is 109,482,240 and BERT-large is 335,141,888, which round to the paper's approximately 110M and 340M. |
