import numpy as np
import cupy as cp
from pathlib import Path
from cuvs.neighbors import brute_force
import datetime

def embedding_validation_test():
    """Simple test using cuVS brute force to validate embeddings."""
    
    # Configuration
    embedding_dir = "/raid/home/ubuntu/filtered-ads-vector-search/amazon_review_embeddings"
    category = "Books"
    n_samples = 1000  # Use 1000 embeddings
    n_queries = 5     # Test 5 queries
    k = 5            # Top 5 neighbors
    
    # Create output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"embedding_validation_results_{timestamp}.txt"
    
    print(f"Loading {n_samples} embeddings from {category}...")
    
    # Load embeddings and texts
    category_path = Path(embedding_dir) / category
    embeddings = np.load(category_path / "embeddings_0.npy")[:n_samples]
    texts = np.load(category_path / "texts_0.npy", allow_pickle=True)[:n_samples]
    
    print(f"Loaded embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Loaded texts: {len(texts)}")
    
    # Convert to GPU and ensure float32 for cuVS
    embeddings_gpu = cp.asarray(embeddings.astype(np.float32))
    
    # Build brute force index
    print("Building brute force index...")
    bf_index = brute_force.build(embeddings_gpu, metric="cosine")
    
    # Select random queries
    np.random.seed(42)
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    
    print("Running validation test and saving results...")
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write("EMBEDDING VALIDATION RESULTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Category: {category}\n")
        f.write(f"  Samples: {n_samples}\n")
        f.write(f"  Queries: {n_queries}\n")
        f.write(f"  Top-K: {k}\n")
        f.write(f"  Embedding shape: {embeddings.shape}\n")
        f.write(f"  Embedding dtype: {embeddings.dtype}\n")
        f.write(f"{'='*80}\n")
        
        for i, query_idx in enumerate(query_indices):
            query_embedding = embeddings_gpu[query_idx:query_idx+1]  # Shape: (1, embedding_dim)
            query_text = texts[query_idx]
            
            f.write(f"\nQUERY {i+1} (Index {query_idx}):\n")
            f.write(f"Text: {query_text}\n")
            f.write(f"-" * 80 + "\n")
            
            # Search for neighbors
            distances, indices = brute_force.search(bf_index, query_embedding, k + 1)  # +1 to include self
            
            # Convert to numpy for easier handling
            distances_np = cp.asnumpy(distances)
            indices_np = cp.asnumpy(indices)
            
            # Handle both 1D and 2D cases
            if distances_np.ndim == 2:
                query_distances = distances_np[0]
                query_indices = indices_np[0]
            else:
                query_distances = distances_np
                query_indices = indices_np
                
            f.write(f"TOP {k} NEIGHBORS:\n")
            neighbor_count = 0
            
            for j in range(len(query_indices)):
                neighbor_idx = query_indices[j]  # numpy scalar
                distance = query_distances[j]     # numpy scalar
                            
                # Skip self
                if neighbor_idx == query_idx:
                    continue
                    
                neighbor_text = texts[neighbor_idx]
                neighbor_count += 1
                
                f.write(f"{neighbor_count}. Index {neighbor_idx} (Distance: {distance:.4f})\n")
                f.write(f"   Text: {neighbor_text}\n\n")
                
                if neighbor_count >= k:
                    break
        
        f.write(f"{'='*80}\n")
        f.write("Test completed successfully!\n")
        f.write(f"{'='*80}\n")
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    embedding_validation_test() 