"""Self-contained deterministic Week 5 speech-recognition foundations lab."""
from collections import Counter
from itertools import product
import math
import numpy as np

CORPUS = [["i", "eat", "apples"], ["i", "eat", "bananas"], ["you", "eat", "apples"]]
BOS, EOS = "<s>", "</s>"

def ngram_counts(sentences):
    unigram, bigram, trigram = Counter(), Counter(), Counter()
    for sentence in sentences:
        seq = [BOS, BOS] + list(sentence) + [EOS]
        unigram.update(sentence)
        bigram.update(zip(seq[1:], seq[2:]))
        trigram.update(zip(seq[:-2], seq[1:-1], seq[2:]))
    return unigram, bigram, trigram

UNIGRAM, BIGRAM, TRIGRAM = ngram_counts(CORPUS)
VOCAB = sorted(set(UNIGRAM) | {EOS})  # BOS is context only.
HISTORY_COUNTS = Counter()
for (history, _word), count in BIGRAM.items():
    HISTORY_COUNTS[history] += count

def bigram_mle(history, word):
    denominator = HISTORY_COUNTS[history]
    if denominator == 0:
        raise ValueError(f"unseen history {history!r}: MLE row is undefined")
    return BIGRAM[(history, word)] / denominator

def bigram_add_one(history, word, alpha=1.0):
    if alpha <= 0:
        raise ValueError("additive smoothing requires positive alpha")
    if word not in VOCAB:
        return 0.0  # Fixed vocabulary: smoothing does not invent an OOV outcome.
    denominator = HISTORY_COUNTS[history] + alpha * len(VOCAB)
    return (BIGRAM[(history, word)] + alpha) / denominator

def utterance_probability(words):
    seq = [BOS] + list(words) + [EOS]
    p = 1.0
    for h, w in zip(seq[:-1], seq[1:]):
        p *= bigram_mle(h, w)
        if p == 0:
            return 0.0
    return p

TRIGRAM_HISTORY = Counter()
for (h1, h2, _word), count in TRIGRAM.items():
    TRIGRAM_HISTORY[(h1,h2)] += count

def trigram_mle(h1, h2, word):
    denominator = TRIGRAM_HISTORY[(h1,h2)]
    if denominator == 0:
        raise ValueError(f"unseen two-token history {(h1,h2)!r}: MLE row is undefined")
    return TRIGRAM[(h1,h2,word)] / denominator

def trigram_probability(words):
    seq = [BOS,BOS] + list(words) + [EOS]
    probability = 1.0
    for h1,h2,word in zip(seq,seq[1:],seq[2:]):
        probability *= trigram_mle(h1,h2,word)
        if probability == 0:
            return 0.0
    return probability

def quantize_1d(values, codebook):
    """Nearest centroid; np.argmin gives deterministic lower-index tie breaking."""
    values, codebook = np.asarray(values, float), np.asarray(codebook, float)
    indices = np.argmin(np.abs(values[:, None] - codebook[None, :]), axis=1)
    return indices, codebook[indices], float(np.mean((values - codebook[indices])**2))

def gaussian_cell_mass(lo, hi, mean, variance):
    scale = math.sqrt(2.0 * variance)
    return 0.5 * (math.erf((hi-mean)/scale) - math.erf((lo-mean)/scale))

def gaussian_pdf(x, mean, variance):
    return math.exp(-0.5*(x-mean)**2/variance) / math.sqrt(2*math.pi*variance)

def forward(pi, transition, emissions):
    alpha = np.asarray(pi, float) * emissions[0]
    for e in emissions[1:]:
        alpha = (alpha @ transition) * e
    return float(alpha.sum())

def enumerate_hmm(pi, transition, emissions):
    n, T = len(pi), len(emissions)
    total = 0.0
    for path in product(range(n), repeat=T):
        p = pi[path[0]] * emissions[0][path[0]]
        for t in range(1, T):
            p *= transition[path[t-1], path[t]] * emissions[t][path[t]]
        total += p
    return total

def acoustic_candidates():
    pi = np.array([1., 0., 0.])
    a = np.array([[.6,.4,0.],[0.,.6,.4],[0.,0.,1.]])
    assert np.allclose(a.sum(1),1) and np.isclose(pi.sum(),1)
    observations = np.array([0.0, 0.1, 0.2])
    means = {"apples": np.array([.2,.4,.6]), "bananas": np.array([0.,.2,.4])}
    variance = .5
    result = {}
    for word, mu in means.items():
        emissions = np.array([[gaussian_pdf(x, m, variance) for m in mu] for x in observations])
        result[word] = forward(pi, a, emissions)
    assert all(math.isclose(forward(pi,a,np.array([[gaussian_pdf(x,m,variance) for m in mu] for x in observations])),
                            enumerate_hmm(pi,a,np.array([[gaussian_pdf(x,m,variance) for m in mu] for x in observations])))
               for mu in means.values())
    return result

def run():
    assert sum(map(len, CORPUS)) == 9
    assert sum(BIGRAM.values()) == 12
    assert not any(w == BOS for (_h,w) in BIGRAM)
    assert BIGRAM[("apples","i")] == 0  # Sentences do not create cross-boundary bigrams.
    assert math.isclose(trigram_probability(["i","eat","apples"]), 1/3)
    assert trigram_probability(["you","eat","bananas"]) == 0
    assert all(math.isclose(sum(trigram_mle(*h,w) for w in VOCAB),1) for h in TRIGRAM_HISTORY)
    assert all(math.isclose(sum(bigram_add_one(h,w) for w in VOCAB),1) for h in [*HISTORY_COUNTS,"unseen"])
    assert bigram_add_one("eat", BOS) == 0 and bigram_add_one("eat", "outside-vocabulary") == 0
    assert math.isclose(bigram_mle("eat", "apples"), 2/3)
    assert math.isclose(utterance_probability(["i","eat","apples"]), 4/9)
    assert math.isclose(TRIGRAM[(BOS,BOS,"you")]/sum(v for (a,b,_),v in TRIGRAM.items() if (a,b)==(BOS,BOS)), 1/3)
    assert math.isclose(utterance_probability(["you","eat","bananas"]), 1/9)
    assert TRIGRAM[("you","eat","bananas")] == 0
    assert math.isclose(sum(utterance_probability(w) for w in product(("i","you"), ("eat",), ("apples","bananas"))), 1)
    assert all(math.isclose(sum(bigram_mle(h,w) for w in VOCAB), 1) for h in HISTORY_COUNTS)
    assert bigram_mle("eat", "you") == 0 and bigram_add_one("eat", "you") > 0
    try:
        bigram_mle("unseen", "apples")
    except ValueError:
        pass
    else:
        raise AssertionError("unseen-history MLE must be undefined")
    print("Corpus lexical tokens:", sum(map(len,CORPUS)))
    print(f"Unigram P(eat): word-only={UNIGRAM['eat']/sum(UNIGRAM.values()):.6f}; with EOS={UNIGRAM['eat']/sum(BIGRAM.values()):.6f}")
    print("Bigram prediction events (BOS once, EOS once per sentence):", sum(BIGRAM.values()))
    print("Prediction vocabulary (BOS excluded):", ", ".join(VOCAB))
    print(f"P(apples | eat) = {bigram_mle('eat','apples'):.6f}")
    print(f"P(i eat apples EOS | BOS) = {utterance_probability(['i','eat','apples']):.6f}")
    print(f"P(you | BOS,BOS) trigram = {TRIGRAM[(BOS,BOS,'you')]/sum(v for (a,b,_),v in TRIGRAM.items() if (a,b)==(BOS,BOS)):.6f}")
    print(f"P(you eat bananas EOS | BOS) bigram = {utterance_probability(['you','eat','bananas']):.6f}; trigram count = {TRIGRAM[('you','eat','bananas')]}")
    probs = [utterance_probability(w) for w in product(("i","you"),("eat",),("apples","bananas"))]
    print(f"P(i eat apples EOS | BOS,BOS) trigram = {trigram_probability(['i','eat','apples']):.6f}")
    print(f"P(you eat bananas EOS | BOS,BOS) trigram = {trigram_probability(['you','eat','bananas']):.6f}")
    print("Four-utterance bigram mass (words + EOS):", f"{sum(probs):.6f}")
    print("P(you | eat), unsmoothed / add-one:", f"{bigram_mle('eat','you'):.6f} / {bigram_add_one('eat','you'):.6f}")
    print("Unseen history MLE: undefined (raised ValueError)")
    lexicon = {"to":[("T","UW")], "two":[("T","UW")], "read":[("R","EH","D"),("R","IY","D")], "red":[("R","EH","D")]}
    print("Toy lexicon entries (ARPAbet-like phones, stress stripped):")
    for word, pronunciations in lexicon.items(): print(f"  {word}: {' | '.join(' '.join(p) for p in pronunciations)}")
    assert lexicon["to"][0] == lexicon["two"][0] and len(lexicon["read"]) == 2
    points = np.array([-1.4,-.9,-.2,.1,.7,1.3]); codebook=np.array([-1.,0.,1.])
    ids, q, distortion = quantize_1d(points, codebook)
    assert np.all(np.diff(codebook)>0) and np.allclose(ids,[0,0,1,1,2,2])
    edges=np.array([-math.inf,-.5,.5,math.inf]); masses=np.array([gaussian_cell_mass(edges[i],edges[i+1],0.,1.) for i in range(3)])
    assert np.isclose(masses.sum(),1.)
    print("VQ feature samples:", np.array2string(points, precision=1))
    print("VQ codebook / assignment indices:", np.array2string(codebook,precision=1), "/", ids.tolist())
    print(f"VQ mean squared quantization distortion (feature^2): {distortion:.6f}")
    print("Illustrative MFCC design: N=3 HMM states; K=2 GMM components/state; D=13 features; M=3 VQ codewords.")
    print("Executed scalar demonstrations: D=1; candidate-HMM emissions use single Gaussians (K=1), without training.")
    print("Gaussian N(0,1) masses in VQ Voronoi cells:", np.array2string(masses,precision=6))
    print("Gaussian density at centroids (feature^-1):", np.array2string(np.array([gaussian_pdf(c,0,1) for c in codebook]),precision=6))
    full=np.ones((3,3))/3
    left=np.array([[.6,.4,0.],[0.,.6,.4],[0.,0.,1.]])
    erg=np.array([[.5,.5,0.],[0.,.5,.5],[.5,0.,.5]])
    assert all(np.allclose(m.sum(axis=1),1) for m in (full,left,erg))
    assert np.all(left[np.tril_indices(3,-1)]==0) and left[2,2]==1
    assert np.all(erg @ erg > 0)  # A positive power establishes finite-chain ergodicity here.
    print("Topology row sums: fully connected =",np.array2string(full.sum(1),precision=1),"; left-to-right =",np.array2string(left.sum(1),precision=1))
    print("Left-to-right forbidden backward transitions: exactly 0; final-state self-loop: 1")
    print("Irreducible, aperiodic counterexample (cycle + self-loops) row sums:",np.array2string(erg.sum(1),precision=1))
    am=acoustic_candidates(); lm={w:bigram_mle("eat",w) for w in am}
    am_winner=max(am,key=am.get); combined={w:am[w]*lm[w] for w in am}; joint_winner=max(combined,key=combined.get)
    assert am_winner=="bananas" and joint_winner=="apples"
    print("Synthetic 3-state continuous-HMM acoustic evidence (feature^-3):", ", ".join(f"{w}={am[w]:.9f}" for w in am))
    print("Acoustic-only winner:",am_winner)
    print("LM priors P(word | eat): " + ", ".join(f"{w}={lm[w]:.6f}" for w in am))
    print("Unnormalized AM × LM scores:",", ".join(f"{w}={combined[w]:.9f}" for w in combined))
    print("Combined winner:",joint_winner)
    print("Acoustic models use the same fixed 3-frame horizon; final states are marginalized, with no exit event.")
    print("Recognition score is a toy candidate comparison; the lexicon above is not compiled into these 3-state word models.")

if __name__ == "__main__": run()
