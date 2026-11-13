# Ebook Discrepancies

The descriptions below reproduce every transcript `possible-error` annotation. They are not adjudicated here.

| # | Week | PDF p. | What the transcript flagged | Status |
|---:|:---:|---:|---|---|
| 1 | 1 | 8 | The displayed outputs in the two conventions are identical after transposing the input matrices, rather than transposes of each other. | unverified (check vs paper) |
| 2 | 1 | 9 | With the row-oriented output defined in Eq. (7), the output projection must multiply on the right, or the head outputs must be transposed. | unverified (check vs paper) |
| 3 | 1 | 10 | The stated matrix shapes require $zW_1$, rather than $W_1z$, in the displayed FFN formula. | unverified (check vs paper) |
| 4 | 1 | 12 | A single-head attention layer can also include an output projection, so $W^O$ is not inherently an extra multi-head cost. | unverified (check vs paper) |
| 5 | 1 | 17 | The displayed value matrix should be $(V^2)^\top=\begin{bmatrix}2&3\\0&4\end{bmatrix}$; the stated output also does not match the shown softmax weights. | unverified (check vs paper) |
| 6 | 2 | 28 | With a single batch value, the displayed training-time normalization gives zero normalized activation and output $\beta$, not the identity. | unverified (check vs paper) |
| 7 | 2 | 29 | Original BERT-large uses post-layer normalization, not Pre-LN. | unverified (check vs paper) |
| 8 | 3 | 37 | The equation uses $h_n[i]$ for predicting $x_i$, whereas the following section says the prediction uses the position $i-1$ hidden state. | unverified (check vs paper) |
| 9 | 3 | 38 | The stated embedding components sum to approximately 31.5 M, not 31.3 M. | unverified (check vs paper) |
| 10 | 3 | 38 | 28.3 M cannot be rounded to 27.6 M. | unverified (check vs paper) |
| 11 | 3 | 38 | The table entries sum to 115.3 M and differ from the preceding calculated component totals. | unverified (check vs paper) |
| 12 | 3 | 42 | The listed components total 114 M, not 117 M. | unverified (check vs paper) |
| 13 | 3 | 44 | The stated total omits the two biases; including them gives 4,726,272 parameters per block. | unverified (check vs paper) |
| 14 | 4 | 46 | The stated forward-pass count conflates candidate tokens with decoder evaluations; $40{,}000^5$ is approximately $10^{23}$ but takes about 3 million years at one billion per second. | unverified (check vs paper) |
| 15 | 4 | 47 | A beam width equal to vocabulary size does not generally make search exhaustive for sequences longer than one token. | unverified (check vs paper) |
| 16 | 4 | 47 | The stated vocabulary contains only A, B and C, so there are no other tokens to receive probability 0.2. | unverified (check vs paper) |
| 17 | 4 | 49 | The equation equates a token $x$ with a logit $u_\ell$; it should identify the token indexed by $\ell$. The zero-temperature limit can have multiple maxima. | unverified (check vs paper) |
| 18 | 4 | 51 | Masking self-attention alone does not remove the input at position $i$ from its output because residual connections retain it. | unverified (check vs paper) |
| 19 | 4 | 55 | The constraint $j\ge i^*$ in the preceding definition prevents the stated edge case $j^*<i^*$. | unverified (check vs paper) |
| 20 | 4 | 56 | The alleged mistake is correct, and the stated fix agrees with it. | unverified (check vs paper) |
| 21 | 4 | 56 | BERT implementations also use attention masks to exclude padding. | unverified (check vs paper) |
