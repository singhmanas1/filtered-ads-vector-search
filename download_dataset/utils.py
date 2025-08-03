import argparse
import pandas as pd
import numpy as np
import tritonclient.grpc
import concurrent.futures
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_NAMES = ["embeddings"]

def get_grpc_client(grpc_port: int):
    """Create and return a Triton GRPC client connected to the specified port."""
    return tritonclient.grpc.InferenceServerClient(url=f"0.0.0.0:{grpc_port}")

def infer_with_grpc(grpc_client, texts, max_batch_size: int, model_name: str, input_type="passage", truncate="END"):
    """
    Calls the GRPC endpoint in batches concurrently to retrieve embeddings.
    
    Parameters:
      - grpc_client (tritonclient.grpc.InferenceServerClient): The GRPC client to use for inference.
      - texts (List[str]): List of input text strings.
      - max_batch_size (int): Maximum number of texts to process per GRPC call.
      - model_name (str): The name of the model to call.
      - input_type (str): The input type to pass to the model (default "passage").
      - truncate (str): The truncation method (default "END").
      
    Returns:
      - A numpy array of concatenated embeddings with shape (num_texts, embedding_dim).
    """
    # Prepare the list of expected outputs
    infer_output = [tritonclient.grpc.InferRequestedOutput(name) for name in OUTPUT_NAMES]
    
    # List to hold our Future objects
    futures = []
    
    # Break texts into batches and send asynchronous requests
    for batch_index, offset in enumerate(range(0, len(texts), max_batch_size)):
        # Extract the current batch of texts
        batch = texts[offset: offset + max_batch_size]
        
        # Prepare the input data as a numpy array of bytes
        text_np = np.array([[text.encode("utf-8")] for text in batch], dtype=np.object_)
        text_input = tritonclient.grpc.InferInput("text", text_np.shape, "BYTES")
        text_input.set_data_from_numpy(text_np)
        infer_input = [text_input]
        
        # Create a Future for this batch
        future = concurrent.futures.Future()
        
        # Define a callback that captures both the result and the batch index
        def callback(result, error, fut=future, index=batch_index):
            if error:
                fut.set_exception(error)
            else:
                # Return both the batch index and result
                fut.set_result((index, result))
        
        # Send the asynchronous request with the callback
        grpc_client.async_infer(
            model_name=model_name,
            inputs=infer_input,
            outputs=infer_output,
            parameters={
                "input_type": input_type,
                "truncate": truncate,
            },
            callback=callback
        )
        futures.append(future)
    
    # Wait for all futures concurrently and collect results as they complete
    # We'll store them in a dict keyed by batch index
    batch_results = {}
    for fut in concurrent.futures.as_completed(futures):
        index, result = fut.result()
        batch_results[index] = result.as_numpy("embeddings")
    
    # Reassemble the results in the correct order
    ordered_results = [batch_results[i] for i in sorted(batch_results)]
    return np.concatenate(ordered_results)

def embed_texts(
    grpc_client,
    texts,
    model_name,
    embedding_folder,
    input_type="passage",
    truncate="END",
    batch_size=10000,
    save_every=10000,
    max_workers=8,
    use_fp16=False,
    n_dimensions=None
):
    """
    Generates embeddings and saves them as embeddings_0.npy, embeddings_1.npy etc.
    
    Args:
        grpc_client: The GRPC client to use for inference
        texts: List of input text strings
        model_name: The name of the model to call
        embedding_folder: The folder to save the embeddings in
        input_type: The input type to pass to the model (default "passage")
        truncate: The truncation method (default "END")
        batch_size: Maximum number of texts to process per GRPC call (default 10000)
        save_every: Number of embeddings to save between saves (default 10000)
        max_workers: Maximum number of threads to use for parallel inference (default 8)
        use_fp16: If True, converts embeddings to float16 before saving
        n_dimensions: If provided, truncates embeddings to first n_dimensions
    """
    os.makedirs(embedding_folder, exist_ok=True)
    num_texts = len(texts)
    batch_indices = list(range(0, num_texts, batch_size))
    file_index = 0
    
    # Variables for accumulating embeddings and texts
    accumulated_embeddings = []
    accumulated_texts = []
    embeddings_processed = 0

    def infer_batch(start_idx):
        batch_texts = texts[start_idx:start_idx+batch_size]
        emb = infer_with_grpc(
            grpc_client,
            batch_texts,
            batch_size,
            model_name,
            input_type=input_type,
            truncate=truncate
        )
        # Truncate dimensions if specified
        if n_dimensions is not None:
            emb = emb[:, :n_dimensions]
        
        # Convert to FP16 if specified
        if use_fp16:
            emb = emb.astype(np.float16)
            
        return (start_idx, emb, batch_texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(infer_batch, idx): idx for idx in batch_indices}
        completed_batches = {}
        next_expected_idx = 0
        
        for fut in as_completed(futures):
            start_idx, emb, batch_texts = fut.result()
            completed_batches[start_idx] = (emb, batch_texts)
            
            # Process completed batches in order
            while next_expected_idx in completed_batches:
                batch_emb, batch_texts = completed_batches.pop(next_expected_idx)
                accumulated_embeddings.append(batch_emb)
                accumulated_texts.append(batch_texts)
                embeddings_processed += len(batch_emb)
                
                # Save when we reach save_every threshold
                if embeddings_processed >= save_every:
                    # Take exactly save_every embeddings and texts
                    combined_embeddings = np.vstack(accumulated_embeddings)
                    combined_texts = np.concatenate(accumulated_texts)
                    embeddings_to_save = combined_embeddings[:save_every]
                    texts_to_save = combined_texts[:save_every]
                    remaining_embeddings = combined_embeddings[save_every:]
                    remaining_texts = combined_texts[save_every:]
                    
                    # Save both embeddings and texts
                    embedding_file = os.path.join(embedding_folder, f"embeddings_{file_index}.npy")
                    text_file = os.path.join(embedding_folder, f"texts_{file_index}.npy")
                    np.save(embedding_file, embeddings_to_save)
                    np.save(text_file, texts_to_save)
                    print(f"Saved {embedding_file} with shape {embeddings_to_save.shape} and dtype {embeddings_to_save.dtype}")
                    print(f"Saved {text_file} with {len(texts_to_save)} texts")
                    
                    file_index += 1
                    
                    # Reset for next chunk
                    if len(remaining_embeddings) > 0:
                        accumulated_embeddings = [remaining_embeddings]
                        accumulated_texts = [remaining_texts]
                        embeddings_processed = len(remaining_embeddings)
                    else:
                        accumulated_embeddings = []
                        accumulated_texts = []
                        embeddings_processed = 0
                
                next_expected_idx += batch_size
        
        # Save any remaining embeddings and texts
        if accumulated_embeddings:
            combined_embeddings = np.vstack(accumulated_embeddings)
            combined_texts = np.concatenate(accumulated_texts)
            embedding_file = os.path.join(embedding_folder, f"embeddings_{file_index}.npy")
            text_file = os.path.join(embedding_folder, f"texts_{file_index}.npy")
            np.save(embedding_file, combined_embeddings)
            np.save(text_file, combined_texts)
            print(f"Saved final {embedding_file} with shape {combined_embeddings.shape} and dtype {combined_embeddings.dtype}")
            print(f"Saved final {text_file} with {len(combined_texts)} texts")

def tokenize_and_count(text: str) -> int:
    """
    Tokenize text by splitting on whitespace and return token count.
    
    Args:
        text: Input text string
        
    Returns:
        Number of tokens (words) in the text
    """
    return len(text.split())

# Example usage:
# save_embeddings_multithreaded_generate_ids(grpc_client, texts, "my_model", "/path/to/embedding_folder")
