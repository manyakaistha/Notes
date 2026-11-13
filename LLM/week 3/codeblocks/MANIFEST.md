# Week 3 executable artifacts

## Code blocks

| File | What it shows | Topic ID |
|---|---|---|
| `01_perplexity.py` / `.txt` | Computes perplexity from mean token cross-entropy and directly from token probabilities; confirms `exp(mean NLL)`. | `w3-language-modelling` |
| `02_chain_rule.py` / `.txt` | Multiplies toy conditional probabilities and compares full-history with a bigram approximation. | `w3-language-modelling` |
| `03_causal_and_shift.py` / `.txt` | Tests causal invariance to changed future tokens and computes shifted next-token cross-entropy in one teacher-forced pass. | `w3-causal-lm` |
| `04_gpt1_parameters.py` / `.txt` | Counts a GPT-1-shaped PyTorch module and formula with tied token/LM weights and explicit bias/LayerNorm conventions. | `w3-gpt1-parameter-count` |
| `05_finetuning_loss.py` / `.txt` | Evaluates a toy task + L2 + weighted L1 fine-tuning loss. | `w3-finetuning` |
| `06_chain_summary.py` / `.txt` | Shows sentence probability and per-token NLL under full-history and bigram toy conditionals. | `w3-language-modelling` |

## Figures

| File | What it shows | Factual caption suggestion |
|---|---|---|
| `assets/01_perplexity_ce.png` | Exponential relationship between cross-entropy and perplexity. | At cross-entropies 0.5, 1, 2, and 3 nats/token, perplexity is respectively 1.65, 2.72, 7.39, and 20.09. |
| `assets/02_chain_rule.png` | Token NLL contributions for the toy sequence under two conditioning schemes. | Full-history conditional probabilities 0.4, 0.6, and 0.5 produce token NLLs 0.916, 0.511, and 0.693 nats, while the toy bigram values 0.4, 0.3, and 0.2 produce larger later losses. |
| `assets/03_causal_mask.png` | Upper-triangular future-token mask for five positions. | The mask blocks the 10 future query-key pairs in a five-token sequence while allowing each position to use its own and preceding positions. |
| `assets/04_gpt1_parameters.png` | GPT-1 component counts under tied-output convention. | The explicit 116,534,784-parameter count comprises 31,480,320 embedding, 28,348,416 attention, 56,669,184 FFN, and 36,864 LayerNorm parameters. |
| `assets/05_task_input_layouts.png` | Serialized task-specific GPT-1 input patterns. | The four layouts distinguish the start, delimiter, and extracted representation positions for classification, entailment, similarity, and multiple choice. |
