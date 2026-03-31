import json
import pandas as pd
from typing import List
from colorama import Fore, Style, init
from tabulate import tabulate

from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from evaluation.metrics import compute_all_metrics
from evaluation.retrieval_eval import retrieval_score_summary

init(autoreset=True)

QUALITATIVE_RUBRIC = """
╔══════════════════════════════════════════════════════════════╗
║           QUALITATIVE ASSESSMENT RUBRIC                      ║
╠══════════════════════════════════════════════════════════════╣
║  Score 4 - Excellent:                                        ║
║    Fully answers the question, factually correct,            ║
║    coherent, well-structured, no hallucination.              ║
║  Score 3 - Good:                                             ║
║    Mostly correct, minor omissions, coherent.                ║
║  Score 2 - Partial:                                          ║
║    Partially answers, some inaccuracies or missing facts.    ║
║  Score 1 - Poor:                                             ║
║    Mostly incorrect, incoherent, or off-topic.               ║
║  Score 0 - Fail:                                             ║
║    Completely wrong or refused to answer.                    ║
╚══════════════════════════════════════════════════════════════╝
"""


class RAGEvaluator:

    def __init__(self, rag_pipeline: RAGPipeline, qa_path: str = "data/qa_pairs.json"):
        self.rag = rag_pipeline
        with open(qa_path, "r") as f:
            self.qa_pairs = json.load(f)
        self.results = []

    def run(self, qualitative_mode: bool = False) -> pd.DataFrame:
        """
        Run evaluation on all Q&A pairs.
        qualitative_mode: if True, prompt human for rubric scores interactively.
        """
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.CYAN + "   RUNNING RAG EVALUATION FRAMEWORK")
        print(Fore.CYAN + "=" * 60)

        if qualitative_mode:
            print(QUALITATIVE_RUBRIC)

        for i, qa in enumerate(self.qa_pairs, 1):
            qid       = qa["id"]
            question  = qa["question"]
            expected  = qa["expected_answer"]
            source_id = qa["source_doc"]

            print(Fore.YELLOW + f"\n[{i}/{len(self.qa_pairs)}] Q{qid}: {question[:70]}...")

            # --- Retrieve ---
            retrieved = self.rag.retriever.retrieve_with_dedup(question)
            context   = self.rag.retriever.format_context(retrieved)

            # --- Generate ---
            generated = self.rag.generator.generate(question, context)
            print(Fore.WHITE + f"  Generated: {generated[:120]}...")

            # --- Quantitative Metrics ---
            metrics = compute_all_metrics(
                generated, expected,
                embedding_model=self.rag.embedding_model
            )

            # --- Retrieval Metrics ---
            ret_metrics = retrieval_score_summary(retrieved, source_id, k=4)

            # --- Qualitative (optional human scoring) ---
            qual_score = None
            if qualitative_mode:
                print(Fore.CYAN   + f"\n  EXPECTED:  {expected}")
                print(Fore.GREEN  + f"  GENERATED: {generated}")
                print(Fore.YELLOW + f"  SOURCES:   {ret_metrics['retrieved_docs']}")
                while True:
                    try:
                        qual_score = int(input(
                            Fore.MAGENTA + "  Your score (0-4): "
                        ))
                        if 0 <= qual_score <= 4:
                            break
                        print("  Enter a number 0-4.")
                    except ValueError:
                        print("  Enter a valid integer.")

            # --- Collect result ---
            self.results.append({
                "id":               qid,
                "question":         question,
                "expected":         expected,
                "generated":        generated,
                "source_doc":       source_id,
                # Quantitative
                "kw_f1":            metrics["keyword_overlap"]["f1"],
                "rouge1_f1":        metrics["rouge"]["rouge1_f1"],
                "rouge2_f1":        metrics["rouge"]["rouge2_f1"],
                "rougeL_f1":        metrics["rouge"]["rougeL_f1"],
                "semantic_sim":     metrics["semantic_similarity"]["cosine_similarity"],
                "semantic_pass":    metrics["semantic_similarity"]["semantic_pass"],
                "fact_coverage":    metrics["key_fact_coverage"]["coverage_rate"],
                "fact_pass":        metrics["key_fact_coverage"]["fact_pass"],
                "composite_score":  metrics["composite_score"],
                "overall_pass":     metrics["overall_pass"],
                # Retrieval
                "precision_at_4":   ret_metrics["precision_at_k"],
                "recall_at_4":      ret_metrics["recall_at_k"],
                "reciprocal_rank":  ret_metrics["reciprocal_rank"],
                "avg_precision":    ret_metrics["average_precision"],
                "retrieval_hit":    ret_metrics["hit"],
                # Qualitative
                "qualitative_score": qual_score
            })

        df = pd.DataFrame(self.results)
        return df

    def print_summary(self, df: pd.DataFrame):
        """Print a rich summary report to the terminal."""
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.CYAN + "   EVALUATION SUMMARY REPORT")
        print(Fore.CYAN + "=" * 60)

        # Per-question table
        table_data = []
        for _, row in df.iterrows():
            status = Fore.GREEN + "PASS" if row["overall_pass"] else Fore.RED + "FAIL"
            hit    = Fore.GREEN + "HIT"  if row["retrieval_hit"] else Fore.RED + "MISS"
            table_data.append([
                f"Q{int(row['id'])}",
                row["question"][:45] + "...",
                f"{row['composite_score']:.3f}",
                f"{row['semantic_sim']:.3f}",
                f"{row['rouge1_f1']:.3f}",
                f"{row['fact_coverage']:.3f}",
                f"{row['recall_at_4']:.1f}",
                f"{row['reciprocal_rank']:.3f}",
                status,
                hit
            ])

        headers = [
            "QID", "Question", "Composite", "SemanticSim",
            "ROUGE-1", "FactCov", "Rec@4", "MRR", "Pass?", "Ret?"
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Aggregate stats
        print(Fore.CYAN + "\n--- AGGREGATE METRICS ---")
        agg = {
            "Total Questions":       len(df),
            "Overall Pass Rate":     f"{df['overall_pass'].mean()*100:.1f}%",
            "Avg Composite Score":   f"{df['composite_score'].mean():.4f}",
            "Avg Semantic Sim":      f"{df['semantic_sim'].mean():.4f}",
            "Avg ROUGE-1 F1":        f"{df['rouge1_f1'].mean():.4f}",
            "Avg ROUGE-2 F1":        f"{df['rouge2_f1'].mean():.4f}",
            "Avg Keyword F1":        f"{df['kw_f1'].mean():.4f}",
            "Avg Fact Coverage":     f"{df['fact_coverage'].mean():.4f}",
            "Retrieval Hit Rate":    f"{df['retrieval_hit'].mean()*100:.1f}%",
            "Avg Reciprocal Rank":   f"{df['reciprocal_rank'].mean():.4f}",
            "Avg Precision@4":       f"{df['precision_at_4'].mean():.4f}",
        }
        for k, v in agg.items():
            color = Fore.GREEN if "Pass" in k or "Hit" in k else Fore.WHITE
            print(f"  {color}{k:<28}{Style.BRIGHT}{v}")

        # Weakest questions
        worst = df.nsmallest(3, "composite_score")[["id", "question", "composite_score"]]
        print(Fore.RED + "\n--- WEAKEST ANSWERS (bottom 3) ---")
        for _, row in worst.iterrows():
            print(f"  Q{int(row['id'])}: {row['question'][:60]}... → {row['composite_score']:.3f}")

        # Best questions
        best = df.nlargest(3, "composite_score")[["id", "question", "composite_score"]]
        print(Fore.GREEN + "\n--- STRONGEST ANSWERS (top 3) ---")
        for _, row in best.iterrows():
            print(f"  Q{int(row['id'])}: {row['question'][:60]}... → {row['composite_score']:.3f}")

    def save_results(self, df: pd.DataFrame, output_dir: str = "evaluation_results"):
        """Save full results to CSV and JSON."""
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        csv_path  = os.path.join(output_dir, "eval_results.csv")
        json_path = os.path.join(output_dir, "eval_results.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(Fore.CYAN + f"\n  Results saved to {csv_path} and {json_path}")