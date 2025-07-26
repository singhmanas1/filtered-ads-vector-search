import json
import argparse
from pathlib import Path
from utils import infer_with_grpc, get_grpc_client, embed_texts

def process_category_files(input_dir: str, output_base_dir: str, use_fp16: bool = False, n_dimensions: int = None):
    """
    Process JSONL files from each category and generate embeddings.
    Args:
        input_dir: Directory containing category JSONL files
        output_base_dir: Base directory where embeddings will be saved by category
        use_fp16: If True, saves embeddings in float16 format
        n_dimensions: If provided, truncates embeddings to first n_dimensions
    """
    grpc_client = get_grpc_client(8001)
    file_counter = 0
    # Process each JSONL file (category)
    for jsonl_file in Path(input_dir).rglob('*.jsonl'):
        category_name = jsonl_file.stem
        print(f"Processing category: {category_name}")
        
        # Create output directory for this category
        #category_output_dir = Path(output_base_dir) / category_name
        category_output_dir = Path(output_base_dir)
        category_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read and process the JSONL file
        texts = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line)
                texts.append(review['text'])

                #If you want to specify number of vectors for each category
                if len(texts)==170000:
                    print("170000 in a category, break")
                    break
        
        print(f"Found {len(texts)} reviews in {category_name}")
        
        # Generate embeddings for this category
        file_counter = embed_texts(
            grpc_client, 
            texts=texts,
            model_name="nvidia_llama_3_2_nv_embedqa_1b_v2",
            embedding_folder=str(category_output_dir),
            batch_size=1000,  # Match the size of each .npy file
            save_every=10000,  # Save every 10k embeddings
            max_workers=224,
            use_fp16=use_fp16,
            n_dimensions=n_dimensions,
            file_counter = file_counter
        )
        file_counter+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings from JSONL files')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory for saving embeddings')
    parser.add_argument('--fp16', action='store_true',
                        help='Convert embeddings to float16')
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Truncate embeddings to first N dimensions')
    
    args = parser.parse_args()
    
    process_category_files(
        args.input_dir,
        args.output_dir,
        use_fp16=args.fp16,
        n_dimensions=args.dimensions
    )