import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from typing import List
import re


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_overlap_score(generated: str, expected: str) -> dict:
    """
    Metric 1: Keyword Overlap (Precision, Recall, F1)
    Measures what fraction of important expected-answer keywords
    appear in the generated answer.
    """
    gen_tokens = set(normalize_text(generated).split())
    exp_tokens = set(normalize_text(expected).split())

    # Remove very common stopwords
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","on",
        "at","by","for","with","about","as","it","its","this","that",
        "these","those","and","or","but","not","from","into","than",
        "then","so","if","when","which","who","what","how","all","also"
    }
    gen_tokens -= stopwords
    exp_tokens -= stopwords

    if not exp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = gen_tokens & exp_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0.0
    recall    = len(common) / len(exp_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "common_keywords": sorted(common)
    }


def rouge_scores(generated: str, expected: str) -> dict:
    """
    Metric 2: ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
    Standard summarization / generation quality metrics.
    ROUGE-1: unigram overlap
    ROUGE-2: bigram overlap
    ROUGE-L: longest common subsequence
    """
    r = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = r.get_scores(generated, expected)[0]
    return {
        "rouge1_f1": round(scores["rouge-1"]["f"], 4),
        "rouge2_f1": round(scores["rouge-2"]["f"], 4),
        "rougeL_f1": round(scores["rouge-l"]["f"], 4),
        "rouge1_precision": round(scores["rouge-1"]["p"], 4),
        "rouge1_recall": round(scores["rouge-1"]["r"], 4),
    }


def semantic_similarity_score(
    generated: str,
    expected: str,
    embedding_model
) -> dict:
    """
    Metric 3: Semantic Similarity (Cosine Similarity on embeddings)
    Captures meaning-level similarity beyond surface word overlap.
    A score >= 0.80 is considered semantically correct.
    """
    gen_emb = embedding_model.embed_single(generated).reshape(1, -1)
    exp_emb = embedding_model.embed_single(expected).reshape(1, -1)
    sim = float(cosine_similarity(gen_emb, exp_emb)[0][0])
    return {
        "cosine_similarity": round(sim, 4),
        "semantic_pass": sim >= 0.75   # threshold for "semantically correct"
    }


def key_fact_coverage(generated: str, expected: str) -> dict:
    """
    Metric 4: Key Fact Coverage (Pass/Fail per key phrase)
    Extracts important noun phrases / numbers / names from the
    expected answer and checks whether the generated answer mentions them.
    """
    # Extract tokens that are likely key facts:
    # numbers, capitalized words (names/proper nouns), hyphenated terms
    fact_pattern = re.compile(
        r"\b(\d[\d,]*\.?\d*%?|\d+s?\b|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*"
        r"|[A-Z]{2,}|[\w]+-[\w]+)\b"
    )
    expected_facts = fact_pattern.findall(expected)
    # Deduplicate while preserving order
    seen = set()
    key_facts = []
    for f in expected_facts:
        if f not in seen and len(f) > 1:
            seen.add(f)
            key_facts.append(f)

    if not key_facts:
        return {"covered": [], "missed": [], "coverage_rate": 1.0}

    gen_lower = generated.lower()
    covered = [f for f in key_facts if f.lower() in gen_lower]
    missed  = [f for f in key_facts if f.lower() not in gen_lower]

    coverage_rate = len(covered) / len(key_facts)
    return {
        "key_facts_expected": key_facts,
        "covered": covered,
        "missed":  missed,
        "coverage_rate": round(coverage_rate, 4),
        "fact_pass": coverage_rate >= 0.6
    }


def length_ratio(generated: str, expected: str) -> dict:
    """
    Metric 5: Answer Length Ratio
    Checks if the generated answer is neither too short nor too verbose.
    Ideal ratio: 0.5 – 2.5x the expected answer length.
    """
    gen_words = len(generated.split())
    exp_words = len(expected.split())
    ratio = gen_words / exp_words if exp_words > 0 else 0.0
    appropriate = 0.4 <= ratio <= 3.0
    return {
        "generated_words": gen_words,
        "expected_words":  exp_words,
        "length_ratio":    round(ratio, 4),
        "length_appropriate": appropriate
    }


def compute_all_metrics(
    generated: str,
    expected: str,
    embedding_model=None
) -> dict:
    """Run all quantitative metrics and return a unified results dict."""
    results = {}
    results["keyword_overlap"]  = keyword_overlap_score(generated, expected)
    results["rouge"]            = rouge_scores(generated, expected)
    results["key_fact_coverage"] = key_fact_coverage(generated, expected)
    results["length_ratio"]     = length_ratio(generated, expected)
    if embedding_model:
        results["semantic_similarity"] = semantic_similarity_score(
            generated, expected, embedding_model
        )
    # Composite pass/fail
    kw_pass   = results["keyword_overlap"]["f1"] >= 0.25
    rouge_pass = results["rouge"]["rouge1_f1"]   >= 0.25
    fact_pass  = results["key_fact_coverage"]["fact_pass"]
    sem_pass   = (results["semantic_similarity"]["semantic_pass"]
                  if embedding_model else True)
    results["overall_pass"] = all([kw_pass, rouge_pass, fact_pass, sem_pass])
    results["composite_score"] = round(
        np.mean([
            results["keyword_overlap"]["f1"],
            results["rouge"]["rouge1_f1"],
            results["key_fact_coverage"]["coverage_rate"],
            results["semantic_similarity"]["cosine_similarity"] if embedding_model else 0.5
        ]), 4
    )
    return results