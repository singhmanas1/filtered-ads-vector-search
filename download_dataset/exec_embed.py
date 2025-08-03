import json
import argparse
from pathlib import Path
import numpy as np
from utils import infer_with_grpc, get_grpc_client, embed_texts,tokenize_and_count

def process_category_files(input_dir: str, output_base_dir: str, use_fp16: bool = False, 
                          n_dimensions: int = None, categories: list = None, max_embeddings: int = None,
                          min_tokens: int = None):
    """
    Process JSONL files from categories and generate embeddings, saving each category separately.
    
    Args:
        input_dir: Directory containing category JSONL files
        output_base_dir: Base directory where embeddings will be saved by category
        use_fp16: If True, saves embeddings in float16 format
        n_dimensions: If provided, truncates embeddings to first n_dimensions
        categories: List of specific categories to process. If None, process all
        max_embeddings: Maximum total embeddings to process across all categories
        min_tokens: Minimum number of tokens required for text to be included
    """
    grpc_client = get_grpc_client(8001)
    
    # Get all JSONL files
    all_files = list(Path(input_dir).rglob('*.jsonl'))
    
    # Filter by categories if specified
    if categories:
        category_set = set(categories)
        filtered_files = [f for f in all_files if f.stem in category_set]
        print(f"Processing specified categories: {categories}")
    else:
        filtered_files = all_files
        print(f"Processing all {len(filtered_files)} categories")

    total_embeddings_processed = 0
    
    # Process each category separately
    for jsonl_file in filtered_files:
        category_name = jsonl_file.stem
        print(f"Processing category: {category_name}")
        
        # Check if we've reached the global limit
        if max_embeddings and total_embeddings_processed >= max_embeddings:
            print(f"Reached global maximum embeddings limit ({max_embeddings}). Stopping.")
            break
        
        # Create output directory for this category
        category_output_dir = Path(output_base_dir) / category_name
        category_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect texts for this category
        category_texts = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Check global limit
                if max_embeddings and (total_embeddings_processed + len(category_texts)) >= max_embeddings:
                    remaining_slots = max_embeddings - total_embeddings_processed
                    print(f"Limiting {category_name} to {remaining_slots} texts due to global limit")
                    break
                    
                review = json.loads(line)
                text = review['text']
                
                # Check minimum token requirement if specified
                if min_tokens is not None:
                    if tokenize_and_count(text) < min_tokens:
                        continue  # Skip this text, don't count toward limit
                
                category_texts.append(text)
        
        if not category_texts:
            print(f"No valid texts found for category {category_name}")
            continue
            
        # Limit category texts if we're approaching global max
        if max_embeddings:
            remaining_global_slots = max_embeddings - total_embeddings_processed
            if len(category_texts) > remaining_global_slots:
                category_texts = category_texts[:remaining_global_slots]
                print(f"Limited {category_name} to {len(category_texts)} texts due to global limit")
        
        print(f"Found {len(category_texts)} valid texts in {category_name}")
        
        # Generate embeddings for this category
        embed_texts(
            grpc_client, 
            texts=category_texts,
            model_name="nvidia_llama_3_2_nv_embedqa_1b_v2",
            embedding_folder=str(category_output_dir),
            batch_size=1000,
            save_every=10000,
            max_workers=224,
            use_fp16=use_fp16,
            n_dimensions=n_dimensions
        )
        
        # Save source mapping for this category (all texts are from same category)
        text_sources = [category_name] * len(category_texts)
        source_file = category_output_dir / "text_sources.npy"
        np.save(source_file, text_sources)
        print(f"Saved text sources mapping to {source_file}")
        
        total_embeddings_processed += len(category_texts)
        print(f"Total embeddings processed so far: {total_embeddings_processed}")
        
        # Check if we've reached the global limit
        if max_embeddings and total_embeddings_processed >= max_embeddings:
            print(f"Reached global maximum embeddings limit ({max_embeddings}). Stopping.")
            break
    
    print(f"Finished processing. Total embeddings: {total_embeddings_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings from JSONL files')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory for saving embeddings by category')
    parser.add_argument('--fp16', action='store_true',
                        help='Convert embeddings to float16')
    parser.add_argument('--dimensions', type=int, default=None,
                        help='Truncate embeddings to first N dimensions')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Specific categories to process (space-separated)')
    parser.add_argument('--max_embeddings', type=int, default=None,
                        help='Maximum total embeddings to process across all categories')
    parser.add_argument('--min_tokens', type=int, default=None,
                        help='Minimum number of tokens required for text inclusion')
    
    args = parser.parse_args()
    
    process_category_files(
        args.input_dir,
        args.output_dir,
        use_fp16=args.fp16,
        n_dimensions=args.dimensions,
        categories=args.categories,
        max_embeddings=args.max_embeddings,
        min_tokens=args.min_tokens
    )