from cuvs.neighbors import cagra
import os
import time
import numpy as np
import pandas as pd
import cupy as cp
import gc
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import argparse
import yaml
import faiss
import numpy as np
from cuvs.neighbors import filters
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
import time
import psutil
import os
import pynvml
from concurrent.futures import wait
from time import perf_counter



from utils_grid_search import (
    # Configuration and Setup
    load_config,
    setup_progress_tracking,
    
    # Data Loading and Processing
    get_algorithm_settings,
    load_vector_data,
    
    # Filtering and Rating
    create_rating_filter,
    
    # Ground Truth and Evaluation
    generate_one_time_ground_truth,
    calculate_recall_with_batching,
    calc_recall,
    
    # Algorithm-specific
    build_hnsw_index
)


def get_memory_usage():
    """Get current memory usage statistics including GPU memory used by this process."""
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    # Initialize pynvml for GPU stats
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    pid = os.getpid()
    gpu_memory_used_by_process = 0
    gpu_memory_total = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_memory_total += pynvml.nvmlDeviceGetMemoryInfo(handle).total
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in procs:
                if p.pid == pid and p.usedGpuMemory is not None:
                    gpu_memory_used_by_process += p.usedGpuMemory
        except pynvml.NVMLError:
            # Some GPUs may not support this query, skip them
            continue

    pynvml.nvmlShutdown()

    return {
        'process_rss_gb': memory_info.rss / (1024**3),  # Resident Set Size in GB
        'process_vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size in GB
        'system_used_gb': system_memory.used / (1024**3),
        'system_available_gb': system_memory.available / (1024**3),
        'system_total_gb': system_memory.total / (1024**3),
        'system_percent_used': system_memory.percent,
        'gpu_memory_used_gb': gpu_memory_used_by_process / (1024**3),
        'gpu_memory_total_gb': gpu_memory_total / (1024**3),
        'gpu_memory_percent_used': (gpu_memory_used_by_process / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
    }

def log_memory_usage(logger, context=""):
    """Log current memory usage with context including GPU memory usage."""
    memory_stats = get_memory_usage()
    logger.info(f"Memory usage {context}:")
    logger.info(f"  Process RSS: {memory_stats['process_rss_gb']:.2f} GB")
    logger.info(f"  Process VMS: {memory_stats['process_vms_gb']:.2f} GB")
    logger.info(f"  System Used: {memory_stats['system_used_gb']:.2f} GB ({memory_stats['system_percent_used']:.1f}%)")
    logger.info(f"  System Available: {memory_stats['system_available_gb']:.2f} GB")
    logger.info(f"  System Total: {memory_stats['system_total_gb']:.2f} GB")
    logger.info(f"  GPU Memory Used: {memory_stats['gpu_memory_used_gb']:.2f} GB ({memory_stats['gpu_memory_percent_used']:.1f}%)")
    logger.info(f"  GPU Memory Total: {memory_stats['gpu_memory_total_gb']:.2f} GB")
    return memory_stats

def generate_cagra_parameter_grid(param_ranges=None, exclude_combinations=None):
    """
    Generate a grid of parameter combinations for CAGRA grid search.
    Allows excluding specific combinations using simple list subtraction.
    
    Args:
        param_ranges: Dictionary with parameter names as keys and lists of values to try.
        exclude_combinations: List of parameter dictionaries to exclude.
                     
    Returns:
        list: List of parameter dictionaries
    """
    # Default parameter ranges if none provided
    if param_ranges is None:

        param_ranges = {
            'intermediate_graph_degree': [64, 80, 96, 112, 128],
            'graph_degree': [32, 48, 64, 80, 96, 112, 128],
            'itopk_size': [64, 80, 96, 112, 128],
            'search_width': [1, 2, 4, 8, 16]
        }
    
    # Ensure all required parameters are in the dictionary
    required_params = ['intermediate_graph_degree', 'graph_degree', 
                       'itopk_size', 'search_width', 'max_search_iterations']
    for param in required_params:
        if param not in param_ranges:
            raise ValueError(f"Missing required parameter range: {param}")
    
    # Generate all parameter combinations
    # Generate all parameter combinations dynamically
    parameter_combinations = []
    
    # Get all parameter names and their values
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[key] for key in param_names]
    
    # Generate all combinations using itertools.product
    import itertools
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        
        # Apply any constraints (like graph_degree <= intermediate_graph_degree)
        if ('graph_degree' in param_dict and 'intermediate_graph_degree' in param_dict and 
            param_dict['graph_degree'] > param_dict['intermediate_graph_degree']):
            continue
        
        parameter_combinations.append(param_dict)
    # Remove excluded combinations if provided
    if exclude_combinations:
        # Convert dictionaries to frozensets of items for hashable comparison
        exclude_set = {frozenset(combo.items()) for combo in exclude_combinations}
        filtered_combinations = []
        
        for combo in parameter_combinations:
            if frozenset(combo.items()) not in exclude_set:
                filtered_combinations.append(combo)
        
        return filtered_combinations
    
    return parameter_combinations

def find_best_configs(results_df, logger=None):
    """
    Find best parameter configurations from results and print them.
    
    Args:
        results_df: DataFrame of results
        
    Returns:
        dict: Dictionary of best configurations
    """
    best_configs = {}
    if not results_df.empty and 'recall' in results_df.columns:
        if not results_df['recall'].isna().all():
            best_configs['best_recall'] = results_df.loc[results_df['recall'].idxmax()]
        
        if 'search_latency_ms' in results_df.columns and not results_df['search_latency_ms'].isna().all():
            best_configs['best_latency'] = results_df.loc[results_df['search_latency_ms'].idxmin()]
        
        if 'build_time_seconds' in results_df.columns and not results_df['build_time_seconds'].isna().all():
            best_configs['best_build_time'] = results_df.loc[results_df['build_time_seconds'].idxmin()]
    
    # Log best configurations (debug level)
    logger.debug("\n=== Best Configurations ===")
    for name, config in best_configs.items():
        logger.debug(f"\n{name}:")
        for key, value in config.items():
            logger.debug(f"  {key}: {value}")
            
    return best_configs

def evaluate_parameter_combination_multithreading(params, vectors, queries, vectors_fp, queries_fp,
                                  vectors_strings=None, queries_strings=None, gt_indices=None,
                                  k=10, refinement=False, rank=False, refinement_ratio=2.0, rank_ratio=2.0,
                                  metric="sqeuclidean", build_algo="nn_descent", quantization_type="fp",
                                  batch_size=None, persistent=False, num_workers=4, filter_obj=None, logger=None):
    """
    Evaluate a specific parameter combination with improved latency and throughput measurements.
    Uses multiprocessing approach similar to the Milvus benchmark to measure performance.
    
    Args:
        params: Parameter configuration for CAGRA
        vectors: Reference vectors for index building
        queries: Query vectors for search
        vectors_fp: Full precision reference vectors (for ground truth or refinement)
        queries_fp: Full precision query vectors (for ground truth or refinement)
        vectors_strings, queries_strings: String representations of vectors (unused)
        gt_indices: Ground truth indices (pre-computed with same filter if used)
        k: Number of nearest neighbors to find
        refinement: Whether to use refinement
        rank: Whether to use re-ranking
        refinement_ratio, rank_ratio: Ratios for refinement/ranking candidate set sizes
        metric: Distance metric to use
        build_algo: Build algorithm for CAGRA
        quantization_type: Type of vector quantization used
        batch_size: Batch size for search operations
        persistent: Whether to use persistent memory allocation
        num_workers: Number of worker threads for concurrent batch processing
        filter_obj: Filter object to apply during search
    """
    import numpy as np
    import cupy as cp
    from multiprocessing import Pool, Manager
    import gc
    
    try:
        # Log testing parameters
        logger.info(f"Testing: {', '.join([f'{k}={v}' for k, v in params.items()])}")
        
        # Log memory usage at start
        initial_memory = log_memory_usage(logger, "at start of parameter evaluation")
        
        # Log if we're using a filter
        if filter_obj is not None:
            logger.info("Using filter for search evaluation")
        
        # Extract parameters
        intermediate_graph_degree = params['intermediate_graph_degree']
        graph_degree = params['graph_degree']
        itopk_size = params['itopk_size']
        search_width = params['search_width']
        max_iterations = params['max_search_iterations']
        
        # Configure build parameters
        build_params = cagra.IndexParams(
            intermediate_graph_degree=intermediate_graph_degree,
            graph_degree=graph_degree,
            metric=metric,
            build_algo=build_algo
        )
        
        # Time the index build
        build_start_time = time.time()
        cagra_index = cagra.build(build_params, vectors)
        build_time = time.time() - build_start_time
        
        # Log memory usage after index build
        post_build_memory = log_memory_usage(logger, "after index build")
        memory_increase_gb = post_build_memory['gpu_memory_percent_used'] - initial_memory['gpu_memory_percent_used']
        logger.info(f"Index build increased memory by {memory_increase_gb:.2f} GB")
        
        # Configure search parameters
        search_params = cagra.SearchParams(itopk_size=itopk_size, search_width=search_width,
                                           persistent=persistent, max_iterations=max_iterations)
        
        # ============== RECALL CALCULATION (SERIAL) ==============
        # Calculate recall first using a batch processing of all queries
        
        # Perform search serially to calculate recall
        start_time = time.time()
        total_search_time = 0
                
        all_indices, total_search_time = calculate_recall_with_batching(queries, cagra_index, search_params, filter_obj, 
                                                                        k, batch_size=batch_size)
        
        # Calculate recall if ground truth available
        recall_score = None
        if gt_indices is not None:
            # Only compare up to n_queries
            recall_score = calc_recall(all_indices, gt_indices)
            logger.info(f"Recall: {recall_score:.4f}")
        
        # ============== THROUGHPUT CALCULATION (PARALLEL) ==============
        # Now measure throughput with multiple processes, similar to Milvus benchmark
        
        # Prepare batches of queries based on different batch sizes
        batch_sizes_to_test = [1, 10, 50] if batch_size is None else [batch_size]
        best_qps = 0
        best_latency = 0
        best_batch_size = 0
        query_time_list = []

        
        for nq in batch_sizes_to_test:
            logger.info(f"Testing throughput with batch_size={nq}, workers={num_workers}...")
            
            # Prepare query batches
            query_batches = []
            for i in range(0, queries.shape[0], nq):
                end = min(i + nq, queries.shape[0])
                query_batches.append(queries[i:end])
            
            # Define non-stop search function (similar to the Milvus example)
            def non_stop_search(cagra_index, search_params, query_batches, run_time, filter_obj, query_time_list):
                # Copy CAGRA index to this process's GPU memory
                queries_processed = 0
                total_time = 0.0
                
                start_time = time.time()
                while True:
                    for query_batch in query_batches:
                        # Convert to GPU
                        query_batch_gpu = cp.asarray(query_batch)
                        
                        # Perform search
                        t1 = time.time()
                        if filter_obj is not None:
                            distances, indices = cagra.search(search_params, cagra_index, query_batch_gpu, k=k, filter=filter_obj)
                        else:
                            distances, indices = cagra.search(search_params, cagra_index, query_batch_gpu, k=k)
                        
                        query_time = time.time() - t1
                        query_time_list.append(query_time)
                        
                        # Update statistics
                        queries_processed += 1  # Count as one batch
                        total_time += query_time
                        
                        # Check if we've run for enough time
                        if total_time >= run_time:
                            return [queries_processed, total_time, query_time_list]
            
            # We can't easily use multiprocessing Pool with CAGRA since the index is in GPU
            # Instead, we'll simulate parallel processing with threads
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            run_time = 10  # 10 seconds per test
            
            t1 = time.time()
            
            # Create thread pool to simulate concurrent queries
            futures = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i in range(num_workers):
                    futures.append(executor.submit(
                        non_stop_search, cagra_index, search_params, query_batches, run_time, filter_obj, query_time_list
                    ))

            wait(futures)

            t2 = time.time()
            
            # Wait for all to complete
            sumq = 0
            sumt = 0.0
            query_time_p99s = []
            result_query_time_list = []
                
            for future in futures:
                try:
                    result = future.result()
                    sumq += result[0]
                    sumt += result[1]
                    query_time_p99s.append(np.percentile(result[2], 99))
                except Exception as e:
                    logger.error(f"Error in worker: {e}")
            
            
            # Calculate metrics - using the same formula as in your code
            #  qps = (sumq * nq) / sumt  # Multiply by batch size and num_workers
            qps = (sumq * nq) / (t2-t1)
            print("sumt " + str(sumt))
            print("(t2-t1) " +  str(t2-t1))
            print("qps " + str(qps))
            
            # latency = sumt / (sumq * nq) * 1000 if sumq > 0 else 0  # ms per query
            latency = sum(query_time_p99s) / len(query_time_p99s) 
            print("latency " + str(latency))
            
            logger.info(f"Batch size {nq}: time usage: {t2-t1:.2f}s, latency: {latency:.2f}ms, qps: {qps:.2f}")
            
            # Track best performance
            if qps > best_qps:
                best_qps = qps
                best_latency = latency
                best_batch_size = nq
            
            # Clean GPU memory
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        
        logger.info(f"Best configuration: batch_size={best_batch_size}, QPS={best_qps:.2f}, latency={best_latency:.2f}ms")
        
        # Return results as a dictionary
        return {
            **params,
            'total_vectors': vectors.shape[0],
            'total_queries': queries.shape[0],
            'build_time_seconds': build_time,
            
            # Parallel execution metrics - using Milvus benchmark methodology
            'best_batch_size': best_batch_size,
            'p99_latency_ms': latency,
            'queries_per_second': qps,
            
            # Result quality metrics
            'recall': recall_score,
            
            # Filter info
            'used_filter': filter_obj is not None,
            
            # Memory usage metrics
            'initial_gpu_memory_percent_used': initial_memory['gpu_memory_percent_used'],
            'post_build_gpu_memory_percent_used': post_build_memory['gpu_memory_percent_used'],
            'memory_increase_build_gb': memory_increase_gb,
        }
        
    except Exception as e:
        logger.error(f"Error in parameter evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# Add this function to generate HNSW parameter grid
def generate_hnsw_parameter_grid(param_ranges=None):
    """Generate a grid of HNSW parameter combinations."""
    # Default parameter ranges if none provided
    print("param ranges ", param_ranges)
    if param_ranges is None:
        param_ranges = {
            'M': [16, 32],                                          # Number of connections per node
            'efConstruction': [32, 64, 128, 256, 512],              # Build-time exploration factor
            'efSearch': [8, 16, 32, 64, 128, 256, 512, 768, 1024],  # Search-time exploration factor
        }
        
    # Ensure all required parameters are in the dictionary
    required_params = ['M', 'efConstruction', 'efSearch']
    for param in required_params:
        if param not in param_ranges:
            raise ValueError(f"Missing required parameter range: {param}")
    
    # Generate parameter combinations
    parameter_combinations = []
    for M in param_ranges['M']:
        for efConstruction in param_ranges['efConstruction']:
            for efSearch in param_ranges['efSearch']:
                parameter_combinations.append({
                    'M': M,
                    'efConstruction': efConstruction,
                    'efSearch': efSearch
                })
    
    return parameter_combinations


# Add this function to evaluate HNSW parameters
def evaluate_hnsw_combination_forloop(params, vectors, queries, vectors_fp, queries_fp,
                             vectors_strings=None, queries_strings=None, gt_indices=None,
                             k=10, quantization_type="fp", batch_size=None, logger=None, filter_obj=None):
    """Evaluate HNSW parameter combination and return results."""
    import time
    import numpy as np
    
    try:
        print("in this function")
        # Log testing parameters
        logger.info(f"Testing HNSW: {', '.join([f'{k}={v}' for k, v in params.items()])}")
        initial_memory = log_memory_usage(logger, "at start of parameter evaluation")
        # Determine batch size if not provided
        if batch_size is None:
            batch_size = min(queries.shape[0], 50)
        
        # Build HNSW index with provided parameters
        params_with_quant = {**params, 'quantization_type': quantization_type}
        hnsw_index, build_time = build_hnsw_index(params_with_quant, vectors)
        sel = faiss.IDSelectorBitmap(filter_obj)
        search_params = faiss.SearchParametersHNSW(sel=sel, efSearch=params['efSearch'])
        # Set up to store all results
        n_queries = queries.shape[0]
        n_candidates = k
        
        # Log memory usage after index build
        post_build_memory = log_memory_usage(logger, "after index build")
        memory_increase_gb = post_build_memory['process_rss_gb'] - initial_memory['process_rss_gb']
        logger.info(f"Index build increased memory by {memory_increase_gb:.2f} GB")

        # Initialize arrays to store results
        all_indices = np.zeros((n_queries, n_candidates), dtype=np.int32)
        all_distances = np.zeros((n_queries, n_candidates), dtype=np.float32)
        
        # Initialize variables for latency tracking
        total_search_time = 0
        search_times = []
        
        # Process queries in batches
        for batch_start in range(0, n_queries, batch_size):
            batch_end = min(batch_start + batch_size, n_queries)
            batch_size_actual = batch_end - batch_start
            
            # Get batch of queries
            batch_queries = queries[batch_start:batch_end]
            
            # Time the search
            batch_start_time = time.time()
            
            # Perform FAISS search
            batch_distances, batch_indices = hnsw_index.search(batch_queries, k=n_candidates,params=search_params)
            
            # Record search time
            batch_search_time = time.time() - batch_start_time
            total_search_time += batch_search_time
            search_times.append(batch_search_time / batch_size_actual * 1000)  # ms per query
            
            # Store results in the overall arrays
            all_indices[batch_start:batch_end, :] = batch_indices
            all_distances[batch_start:batch_end, :] = batch_distances
            # Debug raw search results for first query
            logger.debug(f"Raw search for HNSW params: {params}")
            logger.debug(f"{batch_indices[0]}")
        
        # Calculate average and p99 latency
        avg_query_latency_ms = total_search_time / n_queries * 1000
        p99_query_latency_ms = np.percentile(search_times, 99) if search_times else 0
        
        # Calculate queries per second
        queries_per_second = n_queries / total_search_time
        
        # Calculate recall if we have ground truth
        recall_score = None
        if gt_indices is not None:
            recall_score = calc_recall(all_indices, gt_indices)
        
        logger.info(f"Recall HNSW {recall_score}")

        #Calculate QPS with multithreading - TRIAL

        batch_sizes_to_test = [1, 10, 50] if batch_size is None else [batch_size]
        best_qps = 0
        best_latency = 0
        best_batch_size = 0
        num_workers = 4
        query_time_list = []
        
        for nq in batch_sizes_to_test:
            logger.info(f"Testing throughput with batch_size={nq}, workers={num_workers}...")
            
            # Prepare query batches
            query_batches = []
            for i in range(0, queries.shape[0], nq):
                end = min(i + nq, queries.shape[0])
                query_batches.append(queries[i:end])
            
            # Define non-stop search function (similar to the Milvus example)
            def non_stop_search(hnsw_index, search_params, query_batches, run_time, query_time_list):
                # Copy CAGRA index to this process's GPU memory
                queries_processed = 0
                total_time = 0.0
                
                start_time = time.time()
                while True:
                    for query_batch in query_batches:
                        # Convert to GPU
                        
                        # Perform search
                        t1 = time.time()
                        distances, indices =  hnsw_index.search(batch_queries, k=n_candidates,params=search_params)
                        
                        query_time = time.time() - t1
                        query_time_list.append(query_time)
                        
                        # Update statistics
                        queries_processed += 1  # Count as one batch
                        total_time += query_time
                        
                        # Check if we've run for enough time
                        if total_time >= run_time:
                            return [queries_processed, total_time, query_time_list]
            
            # We can't easily use multiprocessing Pool with CAGRA since the index is in GPU
            # Instead, we'll simulate parallel processing with threads
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            run_time = 10  # 10 seconds per test
            
            t1 = time.time()
            
            # Create thread pool to simulate concurrent queries
            futures = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i in range(num_workers):
                    futures.append(executor.submit(
                        non_stop_search, hnsw_index, search_params, query_batches, run_time, query_time_list
                    ))
            wait(futures)

            t2 = time.time()
                
            # Wait for all to complete
            sumq = 0
            sumt = 0.0
            query_time_p99s = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    sumq += result[0]
                    sumt += result[1]
                    query_time_p99s.append(np.percentile(result[2], 99))
                except Exception as e:
                    logger.error(f"Error in worker: {e}")
            
          
            
            # Calculate metrics - using the same formula as in your code
            # qps = sumq * num_workers / sumt
            qps = (sumq * nq) / (t2-t1)
            print("qps " + str(qps))
            
            # latency = sumt / (sumq * nq) * 1000 if sumq > 0 else 0  # ms per query
            latency = sum(query_time_p99s) / len(query_time_p99s) 
            print("latency " + str(latency))

            
            
            # qps = (sumq * nq * num_workers) / sumt  # Multiply by batch size and num_workers
            # latency = sumt / (sumq * nq) * 1000 if sumq > 0 else 0  # ms per query
            
            logger.info(f"Batch size {nq}: time usage: {t2-t1:.2f}s, latency: {latency:.2f}ms, qps: {qps:.2f}")
            
            # Track best performance
            if qps > best_qps:
                best_qps = qps
                best_latency = latency
                best_batch_size = nq


        # Return results as a dictionary
        return {
            **params,
            'total_vectors': vectors.shape[0],
            'total_queries': queries.shape[0], 
            'build_time_seconds': build_time,
            'search_time_seconds': total_search_time,
            'p99_latency_ms': latency,
            'queries_per_second': qps,
            'batch_size': batch_size,
            'recall': recall_score,
               # Memory usage metrics
            'initial_memory_gb': initial_memory['system_percent_used'],
            'post_build_memory_gb': post_build_memory['system_percent_used'],
            'memory_increase_build_gb': memory_increase_gb,
            }
        
    except Exception as e:
        logger.error(f"Error in HNSW parameter evaluation: {e}")
        raise

def unified_grid_search(vectors=None, queries=None, vectors_fp=None, queries_fp=None,
                      vectors_strings=None, queries_strings=None, gt_indices=None, k=10, 
                      algorithm="cagra", metric=None, build_algo=None, 
                      refinement=False, refinement_ratio=2.0, rank=False, rank_ratio=2.0, 
                      quantization_type="fp16", max_workers=4, quantization_folder=None, 
                      train_indices=None, val_indices=None, batch_size=None, param_ranges=None,
                      logger=None, persistent=False, num_workers=4, rating_ranges=None,
                      run_time_seconds=30, exclude_combinations=None, filter_func=None):
    """
    Unified grid search function that evaluates parameter combinations sequentially,
    but uses the multithreaded throughput evaluation approach.
    
    Args:
        vectors: Original vectors array (only used if quantization_folder is None)
        queries: Original queries array (only used if quantization_folder is None)
        vectors_fp: Full precision vectors (only used if quantization_folder is None)
        queries_fp: Full precision queries (only used if quantization_folder is None)
        vectors_strings: String representations of vectors (optional)
        queries_strings: String representations of queries (optional)
        gt_indices: Ground truth indices for evaluation
        k: Number of nearest neighbors to search for
        algorithm: Which algorithm to use - "cagra" or "hnsw"
        metric: Distance metric (automatically set based on quantization_type if None)
        build_algo: Build algorithm (automatically set based on quantization_type if None)
        refinement: Whether to use refinement (CAGRA only)
        refinement_ratio: Ratio of candidates to retrieve for refinement (CAGRA only)
        rank: Whether to use re-ranking (CAGRA only)
        rank_ratio: Ratio of candidates to retrieve for re-ranking (CAGRA only)
        quantization_type: Type of quantization ('fp', 'sq', 'bq' or 'full_precision', 'scalar', 'binary')
        max_workers: Number of threads to use for parameter evaluation (unused - kept for compatibility)
        quantization_folder: Folder containing pre-quantized vectors
        train_indices: Indices for training vectors to load (required if quantization_folder is provided)
        val_indices: Indices for validation vectors to load (required if quantization_folder is provided)
        batch_size: Batch size for search operations
        param_ranges: Dictionary with parameter ranges to search over
        logger: Logger instance
        persistent: Whether to use persistent memory allocation (CAGRA only)
        num_workers: Number of threads for throughput testing
        rating_ranges: Rating ranges to use for filtering (None, string, list, or dict)
        run_time_seconds: Time to run each throughput test in seconds (default: 30)
        
    Returns:
        pandas.DataFrame: Results for all parameter combinations
    """
    # Set default metric and build algorithm based on algorithm and quantization type
    if algorithm.lower() == "cagra" and (metric is None or build_algo is None):
        default_metric, default_build_algo = get_algorithm_settings(quantization_type)
        metric = metric or default_metric
        build_algo = build_algo or default_build_algo
    
    logger.info("going to load vector")
    vectors, queries, vectors_fp, queries_fp = load_vector_data(
        vectors=vectors,
        queries=queries,
        vectors_fp=vectors_fp,
        queries_fp=queries_fp,
        quantization_type=quantization_type,
        quantization_folder=quantization_folder,
        train_indices=train_indices,
        val_indices=val_indices,
        refinement=refinement,
        rank=rank,
        logger=logger
    )

    # Calculate total vectors count
    total_vectors_count = vectors.shape[0]
    logger.info(f"total vectors count {total_vectors_count}")
    logger.info(f"total queries/val dataset count {queries.shape[0]}")
    
    # Create filter if rating ranges provided
    filter_obj = None
    filtering_percentage = 0  # Default: no filtering
    if rating_ranges is not None:
        logger.info(f"Creating filter for {queries.shape[0]} queries...")
        filter_obj, bitquery = create_rating_filter(
            n_samples=vectors.shape[0],
            n_queries=queries.shape[0],
            valid_ranges=rating_ranges,
            device_id=0,
            verbose=True,
        )
        logger.info("Filter created successfully")
        
        if algorithm.lower() == "hnsw":
            bitquery_numpy = bitquery.get().view(np.uint8) 
            logger.info("Converting filter to numpy ")
            logger.info(f"bit query {bitquery_numpy}")
            logger.info(f"{type(bitquery_numpy)}")
            logger.info(f"{len(bitquery_numpy)}, {bitquery[:10]}")

        # Calculate filtering percentage based on rating distribution
        rating_distribution = {
            '<1.0': 83.63,
            '1.0-2.0': 0.35,
            '2.0-3.0': 0.42,
            '3.0-4.0': 2.42,
            '4.0-5.0': 9.71,
            '=5.0': 3.48
        }
        
        total_percentage = sum(rating_distribution.values())
        
        # Calculate percentage of data included by the filter
        included_percentage = 0
        if isinstance(rating_ranges, str):
            included_percentage = rating_distribution.get(rating_ranges, 0)
        elif isinstance(rating_ranges, list):
            included_percentage = sum(rating_distribution.get(r, 0) for r in rating_ranges)
        elif isinstance(rating_ranges, dict):
            # For dictionary, we'll use the first filter's ranges
            first_filter_key = list(rating_ranges.keys())[0] if rating_ranges else None
            if first_filter_key:
                ranges = rating_ranges[first_filter_key]
                if isinstance(ranges, list):
                    included_percentage = sum(rating_distribution.get(r, 0) for r in ranges)
                else:
                    included_percentage = rating_distribution.get(ranges, 0)
        
        # Calculate percentage filtered out
        filtering_percentage = 100 - (included_percentage / total_percentage * 100)
        
        logger.info(f"Filter excludes approximately {filtering_percentage:.2f}% of data")


    # --- Compute rating_suffix and add vector count and filtering percentage ---
    rating_suffix = ""
    if filter_obj is not None:
        if isinstance(rating_ranges, str):
            rating_suffix = f"_range_{rating_ranges}"
        elif isinstance(rating_ranges, list):
            safe_ranges = "_".join(str(r) for r in rating_ranges)
            rating_suffix = f"_ranges_{safe_ranges}"
        elif isinstance(rating_ranges, dict):
            safe_keys = "_".join(str(k) for k in rating_ranges.keys())
            rating_suffix = f"_ranges_{safe_keys}"

    # Add vector count and filtering percentage to filename
    data_info_suffix = f"_{total_vectors_count}vecs_filter{filtering_percentage:.0f}pct"
    # Generate ground truth if not provided
    if gt_indices is None:
        logger.info("Generating ground truth...")
        gt_indices = generate_one_time_ground_truth(
            train_vectors=vectors_fp,
            val_vectors=queries_fp,
            k=k,
            filter_obj=filter_obj,  # Pass the filter object
            logger=logger
        )
        logger.info("Ground truth generation complete")
    
    # Log algorithm and quantization information
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Quantization type: {quantization_type}")
    logger.info(f"Total vectors: {total_vectors_count:,}")
    logger.info(f"Filtering percentage: {filtering_percentage:.2f}%")

    logger.info(f"Metric: {metric}")
    logger.info(f"Build algorithm: {build_algo}")
    
    # Log rating filter information if provided
    if filter_obj is not None:
        if isinstance(rating_ranges, str):
            logger.info(f"Using rating filter with range: {rating_ranges}")
        elif isinstance(rating_ranges, list):
            logger.info(f"Using rating filter with ranges: {', '.join(rating_ranges)}")
        elif isinstance(rating_ranges, dict):
            logger.info(f"Using rating filters: {', '.join(rating_ranges.keys())}")
    
    # Generate parameter combinations based on algorithm
    if algorithm.lower() == "hnsw":
        parameter_combinations = generate_hnsw_parameter_grid(param_ranges)
    else:  # Default to CAGRA
        parameter_combinations = generate_cagra_parameter_grid(param_ranges, exclude_combinations=exclude_combinations)
    
    logger.info(f"Starting search with {len(parameter_combinations)} combinations sequentially")
    
    # Check GPU memory before starting
    try:
        import cupy as cp
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        logger.info(f"GPU memory before grid search: {(total_mem-free_mem)/(1024**3):.2f}GB / {total_mem/(1024**3):.2f}GB")
    except:
        pass
    
    results = []
    
    # Process parameter combinations sequentially (no ThreadPoolExecutor)
    for i, params in enumerate(parameter_combinations):
        try:
            logger.info(f"Evaluating combination {i+1}/{len(parameter_combinations)}: {params}")
            
            # Call the multithreaded evaluation function for this parameter combination
            if algorithm.lower() == "cagra":
                result = evaluate_parameter_combination_multithreading(
                    params,
                    vectors, queries, vectors_fp, queries_fp,
                    vectors_strings, queries_strings, gt_indices,
                    k, refinement, rank, refinement_ratio, rank_ratio,
                    metric=metric, build_algo=build_algo, quantization_type=quantization_type,
                    batch_size=batch_size, persistent=persistent, num_workers=num_workers,
                    filter_obj=filter_obj, logger=logger
                )
            else:
                # For HNSW (unchanged, could be updated to use multithreading in future)
                result = evaluate_hnsw_combination_forloop(
                    params,
                    vectors_fp if quantization_type in ["fp16", "half_precision"] else vectors,
                    queries_fp if quantization_type in ["fp16", "half_precision"] else queries,
                    vectors_fp, queries_fp,
                    vectors_strings, queries_strings, gt_indices,
                    k, quantization_type, batch_size,logger=logger,filter_obj=bitquery_numpy
                )
                
            results.append(result)
                
            # Add information about filter to result if available
            if filter_obj is not None and algorithm.lower() == "cagra":
                if isinstance(rating_ranges, str):
                    result['rating_filter'] = rating_ranges
                elif isinstance(rating_ranges, list):
                    result['rating_filter'] = ', '.join(rating_ranges)
                elif isinstance(rating_ranges, dict):
                    result['rating_filter'] = 'multiple'
            
            print("results intermed ", result)
            # Save intermediate results
            import pandas as pd
            filename = f'{algorithm}_{quantization_type}{rating_suffix}{data_info_suffix}_grid_search_results_intermediate.csv'
            pd.DataFrame(results).to_csv(filename, index=False)
            
            logger.info(f"Progress: {i+1}/{len(parameter_combinations)} combinations completed")
            
            # Log best results so far
            if len(results) > 0:
                logger.info(f"Current best QPS: {max([r.get('queries_per_second', 0) for r in results]):.2f}")
                logger.info(f"Current best recall: {max([r.get('recall', 0) for r in results]):.4f}")
            
            # Clean GPU memory after each iteration
            try:
                import cupy as cp
                import gc
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            except:
                pass
                
        except Exception as exc:
            error_msg = f"Combination {params} generated an exception: {exc}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            
            results.append({
                **params,
                'algorithm': algorithm,
                'build_time_seconds': None,
                'queries_per_second': None,
                'recall': None,
                'error': str(exc)
            })
            # Also save after exceptions
            import pandas as pd
            filename = f'{algorithm}_{quantization_type}{rating_suffix}{data_info_suffix}_grid_search_results_intermediate.csv'
            pd.DataFrame(results).to_csv(filename, index=False)
    
    # Convert results to DataFrame for analysis
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Find best configurations
    best_configs = find_best_configs(results_df, logger)
    
    # Save final results
    filename = f'{algorithm}_{quantization_type}{rating_suffix}{data_info_suffix}_grid_search_results_new.csv'
    results_df.to_csv(filename, index=False)
    logger.info(f"All results saved to '{filename}'")
    
    # Clean up GPU memory after completion
    try:
        import cupy as cp
        import gc
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        logger.info(f"GPU memory after grid search: {(total_mem-free_mem)/(1024**3):.2f}GB / {total_mem/(1024**3):.2f}GB")
    except:
        pass
    
    return results_df


def main(quantization_folder = "half_precision", 
         quantization_folder_path = "/home/ubuntu/data/home/ubuntu/amazon_ads/embeddings",
         config_path = ".", algo_type="cagra"):
    print("algo type ", algo_type)
    # Systematic evaluation of different data sizes and filtering configurations
    full_quantization_folder_path = os.path.join(quantization_folder_path, quantization_folder)
    # Setup logger once for all runs
    logger = setup_progress_tracking("comprehensive_grid_search_comparison.log")

    # Load configurations from YAML files using the specified config path
    try:
        # Construct full paths to config files
        filter_config_file = os.path.join(config_path, 'filter_config.yaml')
        params_config_file = os.path.join(config_path, 'params_config.yaml')
        
        logger.info(f"Loading configuration files from: {config_path}")
        logger.info(f"Filter config: {filter_config_file}")
        logger.info(f"CAGRA config: {params_config_file}")
        
        # Load filter configuration
        filter_config = load_config(filter_config_file)
        active_filters = filter_config.get('active_filters', ['low_rated', 'high_rated', 'mid_rated'])
        filter_definitions = filter_config['filter_configurations']
        global_filter_settings = filter_config.get('global_filter_settings', {})
        
        # Build filter_configurations list from YAML
        filter_configurations = []
        for filter_name in active_filters:
            if filter_name in filter_definitions:
                filter_def = filter_definitions[filter_name]
                filter_configurations.append((filter_def['ranges'], filter_name))
                logger.info(f"Loaded filter '{filter_name}': {filter_def['description']}")


        params_config = load_config(params_config_file)

        # Load CAGRA parameter configuration
        cagra_params_by_size = params_config['cagra_params_by_size']
        data_sizes = params_config.get('data_sizes', [5010000])
        exclude_combinations = params_config.get('exclude_combinations', [])

        # Load HNSW parameter configuration
        hnsw_params = params_config['hnsw_params']
        
        # Load global settings
        global_settings = params_config.get('global_settings', {})
        num_queries = global_settings.get('num_queries', 10000)
        batch_size = global_settings.get('batch_size', 100)
        num_workers_params = global_settings.get('num_workers_params', 1)
        num_workers_throughput = global_settings.get('num_workers_throughput', 2)
        persistent = global_settings.get('persistent', False)
        run_time_seconds = global_settings.get('run_time_seconds', 30)
        
        # Load results settings
        results_settings = params_config.get('results_settings', {})
        results_dir = results_settings.get('results_dir', 'comprehensive_comparison_results-test')
        save_intermediate = results_settings.get('save_intermediate', True)
        cleanup_pause_seconds = results_settings.get('cleanup_pause_seconds', 30)
        
        logger.info(f"Loaded {len(filter_configurations)} filter configurations")
        if algo_type == "hnsw":
            logger.info(f"Loaded CAGRA parameters for {len(cagra_params_by_size)} data sizes")
        logger.info(f"Data sizes to evaluate: {data_sizes}")
        
    except Exception as e:
        logger.error(f"Error loading configuration files from {config_path}: {e}")
        
    # Results tracking
    all_results = []
    total_runs = len(data_sizes) * len(filter_configurations)
    run_counter = 0

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"{data_sizes}")
    # Main loop for data sizes  
    for data_size in data_sizes:
        logger.info(f"========= Processing dataset with {data_size:,} vectors =========")
        
        # Generate train/val indices for this data size
        all_indices = np.arange(data_size)
        print("num queries",num_queries)
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=num_queries,
            random_state=42  # Same random seed for reproducibility
        )
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
        
        logger.info(f"Training set size: {len(train_indices):,}")
        logger.info(f"Validation set size: {len(val_indices):,}")
        
        # Get the appropriate parameter grid for this data size
        cagra_params = cagra_params_by_size[data_size]

        if algo_type.lower() == "cagra":
                algo_params = cagra_params
        else: algo_params = hnsw_params
        # Inner loop for filter configurations
        for filter_ranges, filter_name in filter_configurations:
            run_counter += 1
            logger.info(f"--- Run {run_counter}/{total_runs}: {data_size:,} vectors with filter: {filter_name} ---")
            
            # Log memory usage before run
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            logger.info(f"GPU memory before run: {(total_mem-free_mem)/(1024**3):.2f}GB / {total_mem/(1024**3):.2f}GB")
            
            
            try:
                # Run grid search with this configuration - using the new unified_grid_search function
                start_time = time.time()
                results = unified_grid_search(
                    algorithm=algo_type,
                    quantization_type="fp16",
                    quantization_folder=full_quantization_folder_path,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    k=10,
                    param_ranges=algo_params,
                    #  max_workers=num_workers_params,  # This is now unused but kept for compatibility
                    logger=logger,
                    batch_size=batch_size,
                    persistent=persistent,
                    num_workers=num_workers_params,
                    rating_ranges=filter_ranges,  # Apply the current filter configuration
                    run_time_seconds=run_time_seconds,  # New parameter for throughput test duration
                    exclude_combinations=exclude_combinations,
                )
                
                print("Results ,", results)
                # Calculate run time
                run_time = time.time() - start_time
                
                # Find best results
                best_recall = results['recall'].max() if 'recall' in results and not results['recall'].empty else None
                best_qps = results['queries_per_second'].max() if 'queries_per_second' in results and not results['queries_per_second'].empty else None
                
                # Get parameter settings for best QPS
                best_qps_params = {}
                if 'queries_per_second' in results and not results['queries_per_second'].empty:
                    max_qps_idx = results['queries_per_second'].idxmax()
                    best_row = results.iloc[max_qps_idx]
                    best_qps_recall = best_row['recall'] if 'recall' in best_row else None
                    
                    # Extract parameter values that gave best QPS
                    for param in ['intermediate_graph_degree', 'graph_degree', 'itopk_size', 'search_width']:
                        if param in best_row:
                            best_qps_params[param] = best_row[param]

                # Track results with additional metrics
                run_summary = {
                    'data_size': data_size,
                    'filter_name': filter_name,
                    'filter_ranges': str(filter_ranges),
                    'run_time_minutes': run_time / 60,
                    'best_recall': best_recall,
                    'best_qps': best_qps,
                    'best_qps_params': str(best_qps_params),
                    'parameters_tested': len(results),
                    'output_file': f"{results_dir}/results_{data_size}vecs_{filter_name}.csv"
                }
                all_results.append(run_summary)
                
                # Save detailed results
                results.to_csv(f"{results_dir}/results_{data_size}vecs_{filter_name}.csv", index=False)
                
                # Save overall progress summary
                import pandas as pd
                pd.DataFrame(all_results).to_csv(f"{results_dir}/all_runs_summary-2.csv", index=False)
                
                logger.info(f"Completed run in {run_time/60:.2f} minutes")
                logger.info(f"Best recall: {best_recall}")
                logger.info(f"Best QPS: {best_qps}")
                
            except Exception as e:
                logger.error(f"Error in run with {data_size} vectors and filter {filter_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                all_results.append({
                    'data_size': data_size,
                    'filter_name': filter_name,
                    'filter_ranges': str(filter_ranges),
                    'error': str(e)
                })
                pd.DataFrame(all_results).to_csv(f"{results_dir}/all_runs_summary-2.csv", index=False)
            
            # Clean up after run
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            logger.info(f"GPU memory after run: {(total_mem-free_mem)/(1024**3):.2f}GB / {total_mem/(1024**3):.2f}GB")
            logger.info(f"Waiting {cleanup_pause_seconds} seconds before next run to ensure memory cleanup...")
            time.sleep(cleanup_pause_seconds)  # Configurable pause between runs

    # Final summary
    logger.info("========= All runs completed =========")
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Results saved in {results_dir}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CUVS benchmark with configurable quantization folder')

    parser.add_argument(
        '--quantization-folder', 
        type=str, 
        required=True,
        help='Path to the quantization folder (e.g., "half_precision", "full_precision")'
    )
    parser.add_argument(
        '--quantization-folder-path',
        type=str,
        default='.',
        help='Base path where quantization folders are located (default: current directory)'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='.',
        help='Path where configuration files (filter_config.yaml, params_config.yaml) are located (default: current directory)'
    )
    parser.add_argument(
        '--algo-type',
        type=str,
        default='cagra',
        help='Algorithm Type - cagra or hnsw'
    )
    args = parser.parse_args()
    main(args.quantization_folder, args.quantization_folder_path, args.config_path, args.algo_type)