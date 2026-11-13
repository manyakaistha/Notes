#!/usr/bin/env python3
"""Create a review-only keyword routing draft for the LLM question inventory."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "sources/exam-papers/question-inventory.json"
TOPICS_PATH = ROOT / "sources/topics.json"
OUTPUT_PATH = ROOT / "sources/exam-papers/routing-draft.json"
REPORT_PATH = ROOT / "sources/exam-papers/routing-draft-report.txt"

# Weighted signals beyond the canonical syllabus keywords. Longer, more
# discriminative phrases receive greater weight than generic single terms.
EXTRA_TOPIC_TERMS: dict[str, list[str]] = {
    "w1-seq2seq-bottleneck": ["sequence to sequence", "seq2seq", "context vector", "vanishing gradient", "fixed length"],
    "w1-bahdanau-attention": ["bahdanau", "additive attention", "alignment score", "alignment matrix"],
    "w1-self-attention-qkv": ["self attention", "query key value", "q/k/v", "w^q", "w^k", "w^v", "w_q", "w_k", "w_v"],
    "w1-scaled-dot-product": ["scaled dot product", "sqrt(d_k)", "sqrt d_k", "d_k", "attention logits", "gradient saturation"],
    "w1-multi-head-attention": ["multi head attention", "multiheaded attention", "attention head", "w^o", "head concatenation"],
    "w1-ffn-encoder-stack": ["position wise feed forward", "feed forward network", "encoder stack", "relu", "residual connection"],
    "w1-encoder-parameter-count": ["parameter count", "trainable parameters", "number of parameters"],
    "w2-teacher-forcing": ["teacher forcing", "ground truth token", "shifted right", "exposure bias", "scheduled sampling"],
    "w2-masked-self-attention": ["masked self attention", "look ahead mask", "causal mask", "future token"],
    "w2-cross-attention": ["cross attention", "encoder decoder attention", "encoder output", "source sequence"],
    "w2-positional-encoding": ["positional encoding", "positional embedding", "position encoding", "sinusoidal", "sequence order"],
    "w2-residual-layernorm": ["layer norm", "layer normalization", "batch norm", "batch normalization", "add & norm", "skip connection", "residual"],
    "w2-training-tricks": ["label smoothing", "warm up", "warmup", "learning rate schedule", "attention dropout"],
    "w2-full-transformer": ["full transformer", "transformer architecture", "encoder decoder architecture"],
    "w3-language-modelling": ["language modelling", "language modeling", "perplexity", "chain rule", "negative log likelihood", "next token prediction"],
    "w3-model-parameterisations": ["autoregressive factorization", "joint probability", "conditional model", "unconditional model"],
    "w3-causal-lm": ["causal language model", "causal language modelling", "decoder only", "clm", "causal transformer"],
    "w3-gpt1-architecture": ["gpt-1", "gpt 1", "bookscorpus", "pretrain fine tune", "pre-training and fine-tuning"],
    "w3-gpt1-parameter-count": ["117m", "117 million", "gpt-1 parameter"],
    "w3-pretraining-objective": ["pretraining objective", "pre-training objective", "maximum likelihood", "teacher forcing"],
    "w3-finetuning": ["fine tuning", "fine-tuning", "downstream task", "classification head", "transfer learning"],
    "w3-task-input-formats": ["sentiment classification", "textual entailment", "natural language inference", "multiple choice", "text generation"],
    "w4-exhaustive-greedy": ["exhaustive search", "greedy search", "locally optimal", "globally optimal", "decoding search space"],
    "w4-beam-search": ["beam search", "beam width", "beam size", "length normalization", "length normalisation"],
    "w4-topk-temperature": ["top-k", "top k", "temperature scaling", "sampling temperature", "stochastic decoding"],
    "w4-topp-nucleus": ["top-p", "top p", "nucleus sampling", "cumulative probability", "repetition penalty"],
    "w4-bert-mlm": ["bert", "masked language model", "masked language modeling", "masked language modelling", "mlm", "[mask]", "80/10/10"],
    "w4-bert-nsp-inputs": ["next sentence prediction", "nsp", "[cls]", "[sep]", "segment embedding", "sentence pair"],
    "w4-bert-architecture-params": ["bert-base", "bert-large", "110m", "340m", "bert parameters"],
    "w4-bert-finetuning": ["bert fine tuning", "bert fine-tuning", "squad", "span prediction", "feature extractor", "gpt vs bert"],
}

# These indicate concepts intended for later syllabus weeks or special review.
ARCHIVE_SIGNALS: dict[str, list[str]] = {
    "w5-tokenization": ["tokenizer", "tokenization", "tokenisation", "byte pair encoding", "bpe", "wordpiece", "sentencepiece", "sentence piece", "unigram tokenizer", "subword", "out-of-vocabulary", "oov token"],
    "w7-model-families": ["bart", "t5", "text-to-text transfer transformer", "gpt-2", "gpt 2", "encoder-decoder llm", "denoising pretraining", "span corruption"],
    "w8-scaling-laws": ["scaling law", "scaling laws", "chinchilla", "compute optimal", "compute-optimal", "data scaling", "training data size", "model scaling", "parameter-data tradeoff"],
    "w11-efficient-long-context": ["long context", "long sequence", "flashattention", "flash attention", "sparse attention", "linear attention", "kv cache", "key-value cache", "attention memory complexity"],
    "w12-position-methods": ["rope", "rotary position", "rotary positional", "alibi", "attention with linear biases", "nope", "relative positional encoding", "relative position embedding"],
    "later-alignment-and-adaptation": ["rlhf", "reinforcement learning from human feedback", "instruction tuning", "instruction following", "prompt engineering", "prompting", "in-context learning", "few-shot prompting", "lora", "low-rank adaptation", "rag", "retrieval augmented generation", "retrieval-augmented generation", "dpo", "constitutional ai", "preference model", "chain of thought", "tool use", "agentic"],
    "non_llm": ["photosynthesis", "cellular respiration", "mitochondria", "organic chemistry", "calculus", "database normalization", "operating system", "computer network", "linear regression", "support vector machine", "decision tree", "convolutional neural network", "image segmentation", "sorting algorithm", "database query", "thermodynamics"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\\sqrt{d_k}", " sqrt d_k ").replace("√d_k", " sqrt d_k ").replace("√dₖ", " sqrt d_k ")
    text = text.replace("\\text", " ").replace("\\mathrm", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("^", " ").replace("_", " ")
    text = re.sub(r"[^\w\[\]/+-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize common mathematical/model spellings while retaining canonical
    # tokens such as [CLS] and [MASK].
    text = re.sub(r"\bw\s+q\b", "w q", text)
    return text


def variants(term: str) -> list[str]:
    n = normalize(term)
    out = {n}
    out.add(n.replace("-", " "))
    out.add(n.replace(" ", "-"))
    out.add(n.replace(" ", ""))
    # Terms containing TeX projection names normalize to e.g. "w q".
    return sorted((x for x in out if x), key=len, reverse=True)


def term_weight(term: str) -> float:
    words = re.findall(r"[\w]+", normalize(term))
    if len(words) >= 3:
        return 3.0
    if len(words) == 2:
        return 2.0
    return 1.0


def match_terms(text: str, terms: list[str]) -> tuple[float, list[str]]:
    matched: list[str] = []
    score = 0.0
    for term in terms:
        if any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text) for v in variants(term)):
            matched.append(term)
            score += term_weight(term)
    return score, matched


def flatten_keywords(topic: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(topic.get("keywords", []) + EXTRA_TOPIC_TERMS.get(topic["id"], [])))


def route(question: dict[str, Any], topics_by_week: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pieces = [question.get("stem_text", "") or ""]
    option_texts = []
    image_refs = bool(question.get("stem_images"))
    for option in question.get("options", []) or []:
        option_texts.append(option.get("text", "") or "")
        image_refs = image_refs or bool(option.get("images"))
    pieces.extend(option_texts)
    raw_text = " ".join(pieces)
    text = normalize(raw_text)
    visible_chars = len(re.sub(r"[^a-z0-9]+", "", text))
    image_only = bool(question.get("image_only")) or (image_refs and visible_chars < 24)

    topic_hits: list[tuple[float, str, str, list[str]]] = []
    week_scores: dict[int, float] = {}
    for week, topics in topics_by_week.items():
        w = int(week)
        for topic in topics:
            score, matched = match_terms(text, flatten_keywords(topic))
            if score:
                # A small phrase-density normalization prevents long keyword
                # lists from dominating solely because of list size.
                topic_hits.append((score, topic["id"], week, matched))
                week_scores[w] = week_scores.get(w, 0.0) + score

    archive_hits: list[str] = []
    archive_score = 0.0
    for signal, terms in ARCHIVE_SIGNALS.items():
        score, matched = match_terms(text, terms)
        if matched:
            archive_hits.extend(f"{signal}:{m}" for m in matched)
            archive_score += score

    # A no-hit question still needs a review bucket. This is deliberately
    # marked as unclassified rather than asserting that its subject is known.
    if not topic_hits and not archive_hits:
        archive_hits.append("non_llm:unclassified")

    ordered_weeks = sorted(week_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    best_week, best_score = ordered_weeks[0] if ordered_weeks else (None, 0.0)
    competing = len(ordered_weeks) > 1 and ordered_weeks[1][1] >= best_score * 0.72
    # Archive is the destination when later-week signals dominate or no W1-W4
    # topic signal exists. A direct non-LLM signal also routes to archive.
    if archive_score > best_score or best_week is None:
        candidate: int | str = "archive"
        topic_hits = [] if archive_score > best_score else topic_hits
        denominator = archive_score + best_score
        confidence = archive_score / denominator if denominator else 0.0
        if archive_score and best_score and archive_score > best_score:
            evidence = archive_hits[:]
        else:
            evidence = [m for _, _, _, ms in topic_hits for m in ms]
    else:
        candidate = best_week
        denominator = sum(week_scores.values())
        confidence = best_score / denominator if denominator else 0.0
        evidence = [m for _, _, w, ms in topic_hits if int(w) == best_week for m in ms]

    ranked_topics = sorted(topic_hits, key=lambda hit: (-hit[0], hit[1]))
    topic_ids = list(dict.fromkeys(hit[1] for hit in ranked_topics[:3]))
    if candidate == "archive" and archive_score >= best_score:
        topic_ids = []
    low_confidence = confidence < 0.5 or image_only or not (topic_hits or archive_hits) or competing
    return {
        "id": question["id"],
        "candidate_week": candidate,
        "topic_ids": topic_ids,
        "archive_signals": archive_hits,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "low_confidence": bool(low_confidence),
        "image_only": image_only,
        "evidence": list(dict.fromkeys(evidence)),
    }


def main() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    topics_doc = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    topics_by_week = {str(k): v["topics"] for k, v in topics_doc["weeks"].items()}
    questions = inventory["questions"]
    records = [route(q, topics_by_week) for q in questions]
    result = {"generated_by": "LLM/scripts/route_draft.py", "records": records}
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    week_counts = Counter(str(r["candidate_week"]) for r in records)
    topic_counts = Counter(tid for r in records for tid in r["topic_ids"])
    low_count = sum(r["low_confidence"] for r in records)
    image_count = sum(r["image_only"] for r in records)
    lines = [
        "FIRST-PASS KEYWORD ROUTING DRAFT — HUMAN REVIEW REQUIRED",
        "This heuristic output is not a correctness claim; keyword overlap can misroute questions.",
        f"Total records: {len(records)}",
        "Counts by candidate_week: " + ", ".join(f"{k}={week_counts.get(k, 0)}" for k in ["1", "2", "3", "4", "archive"]),
        f"Low-confidence count: {low_count}",
        f"Image-only count: {image_count}",
        "Counts per topic id: " + (", ".join(f"{k}={v}" for k, v in sorted(topic_counts.items())) or "(none)"),
        "",
        "10 sample records:",
    ]
    for rec in records[:10]:
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
