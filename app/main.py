from services.rag_service import RagChain
import argparse
import warnings

warnings.filterwarnings("ignore", module="langsmith")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question for RAG')
    parser.add_argument('--question', type=str, help='Ask a question to the RAG service', required=True)
    args = parser.parse_args()

    response = RagChain() \
        .with_question("Er partiet Venstre positive til private helsetjenester?") \
        .with_llm("deepseek-r1:8b") \
        .with_multi_querying() \
        .run()
    