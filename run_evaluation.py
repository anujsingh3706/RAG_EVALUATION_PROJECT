import argparse
from colorama import Fore, init
from src.rag_pipeline import RAGPipeline
from evaluation.evaluator import RAGEvaluator

init(autoreset=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Evaluation Framework"
    )
    parser.add_argument(
        "--qualitative", action="store_true",
        help="Enable interactive human qualitative scoring (0-4 rubric)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of FAISS vector index from scratch"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results",
        help="Output directory for results CSV/JSON"
    )
    args = parser.parse_args()

    print(Fore.CYAN + "\n" + "=" * 60)
    print(Fore.CYAN + "   RAG EVALUATION FRAMEWORK — QUANTUM COMPUTING DOMAIN")
    print(Fore.CYAN + "=" * 60)

    # Initialize pipeline
    rag = RAGPipeline(
        docs_dir="data/documents",
        top_k=4,
        force_rebuild=args.rebuild
    )

    # Run evaluator
    evaluator = RAGEvaluator(rag, qa_path="data/qa_pairs.json")
    df = evaluator.run(qualitative_mode=args.qualitative)

    # Print summary
    evaluator.print_summary(df)

    # Save results
    evaluator.save_results(df, output_dir=args.output)

    print(Fore.GREEN + "\n Evaluation complete!\n")

if __name__ == "__main__":
    main()