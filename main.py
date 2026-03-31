from colorama import Fore, Style, init
from src.rag_pipeline import RAGPipeline

init(autoreset=True)

def print_banner():
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "   QUANTUM COMPUTING HISTORY — RAG SYSTEM")
    print(Fore.CYAN + "   Domain: History of Quantum Computing")
    print(Fore.CYAN + "=" * 60)

def print_result(result: dict):
    print(Fore.GREEN + "\n ANSWER:")
    print(Style.BRIGHT + result["answer"])
    print(Fore.YELLOW + "\n SOURCES USED:")
    for s in result["sources"]:
        print(f"   • {s['title']} (score: {s['score']:.4f})")
    print(Fore.CYAN + "-" * 60)

def main():
    print_banner()
    print(Fore.WHITE + "\nInitializing RAG Pipeline...")
    rag = RAGPipeline(docs_dir="data/documents", top_k=4)

    print(Fore.CYAN + "\nRAG System ready! Type your question or 'quit' to exit.\n")

    while True:
        question = input(Fore.WHITE + "Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print(Fore.CYAN + "Goodbye!")
            break
        if not question:
            continue
        result = rag.query(question)
        print_result(result)

if __name__ == "__main__":
    main()