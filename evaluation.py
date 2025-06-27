import argparse
import jsonlines
from tqdm import tqdm
from rag_system import RAGSystem  # Importing the RAGSystem class from rag.py
import os


import time
from google.api_core.exceptions import ResourceExhausted

def generate_sql_with_retry(rag_system, arabic_text, retries=5, default_delay=60):
    """Generate SQL using the RAG system with retry logic in case of quota exhaustion (429)."""
    attempt = 0
    while attempt < retries:
        try:
            # Generate SQL using RAG system
            generated_sql = rag_system.query(arabic_text)
            return generated_sql  # If successful, return the result
        except ResourceExhausted as e:
            attempt += 1
            # Extract retry delay from the error message or use default delay
            try:
                # Check if the error message contains retry delay information
                delay = e.retry_info.retry_delay.total_seconds() if e.retry_info else default_delay
            except AttributeError:
                # If no retry_info available, use default delay
                delay = default_delay
            
            print(f"Quota exceeded. Retrying after {delay} seconds... (Attempt {attempt}/{retries})")
            time.sleep(delay)  # Wait for the specified delay before retrying
    print("Exceeded retry attempts. Please check your API quota.")
    return None  # Return None if all retries failed





def evaluate_rag_system(test_data, rag_system):
    """Evaluates the RAG system using the provided test data."""
    total_examples = len(test_data[:100])  # Evaluate only the first 100 examples
    correct_matches = 0  # Initialize the correct matches counter

    for i, example in tqdm(enumerate(test_data[:100]), total=min(100, len(test_data)), desc="Evaluating examples"):
        arabic_text = example['arabic']
        target_sql = example['query']  # Assuming 'query' contains the correct SQL query

        # Generate SQL using the RAG system with retry logic
        generated_sql_from_rag = generate_sql_with_retry(rag_system, arabic_text)

        if generated_sql_from_rag is not None:
            # Compare generated SQL with the target SQL
            exact_match = generated_sql_from_rag.strip() == target_sql.strip()

            if exact_match:
                correct_matches += 1  # Increment the correct match counter

            # Output the result for each example
            print(f"\nExample {i + 1}:")
            print(f"Arabic Text: {arabic_text}")
            print(f"Generated SQL (from RAG): {generated_sql_from_rag}")
            print(f"Target SQL: {target_sql}")
            print(f"Exact Match: {exact_match}")
        else:
            print(f"Failed to generate SQL for example {i + 1} after retries.")

    # Calculate and print the overall percentage of correct matches
    if total_examples > 0:
        percentage = (correct_matches / total_examples) * 100
        print(f"\nOverall Exact Match Percentage: {percentage:.2f}%")
    else:
        print("No examples were evaluated.")

def main():
    """Main function to run the RAG system and evaluate results."""
    parser = argparse.ArgumentParser(description="RAG System for SQL Queries")
    parser.add_argument(
        "--data", type=str, default="AR_spider.jsonl", help="Path to JSONL data file"
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--db_id", type=str, help="Filter results by database ID")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--question", type=str, help="Question to answer (non-interactive mode)")

    args = parser.parse_args()

    # Load Gemini API key from environment variable or use default
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBqSqj418HluZowR58hDrmwOmnLf_7x7cA")

    # Initialize the RAG system
    rag_system = RAGSystem(gemini_api_key=gemini_api_key, model_name="silma-ai/silma-embeddding-matryoshka-0.1")

    # Load the test data from the JSONL file
    with jsonlines.open(args.data) as f:
        test_data = [obj for obj in f]

    # Build or load the index
    rag_system.build_index(args.data, force_rebuild=args.rebuild)

    # Evaluate using the RAG system
    evaluate_rag_system(test_data, rag_system)

if __name__ == "__main__":
    main()
