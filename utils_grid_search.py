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

import os
import requests
import json
import time
import threading
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import List
from urllib.parse import urlparse
import faiss
from tqdm import tqdm
#from datasets import load_dataset

#from minio import Minio
#from minio import Minio, S3Error
from urllib3 import PoolManager
#from pymilvus import MilvusClient, DataType
import cupy as cp
import h5py
import os
import tempfile
import time
import urllib
import numpy as np

import psutil
import pynvml
import time
import threading
from functools import wraps
from collections import defaultdict
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import nest_asyncio

import cuvs

# Importing 'calc_truth' from generate_groundtruth (from cuvs_bench)
import sys
import os
import faiss

## Check the quality of the prediction (recall)
def calc_recall(found_indices, ground_truth):
    found_indices = cp.asarray(found_indices)
    bs, k = found_indices.shape
    if bs != ground_truth.shape[0]:
        raise RuntimeError(
            "Batch sizes do not match {} vs {}".format(
                bs, ground_truth.shape[0]
            )
        )
    if k > ground_truth.shape[1]:
        raise RuntimeError(
            "Not enough indices in the ground truth ({} > {})".format(
                k, ground_truth.shape[1]
            )
        )
    n = 0
    # Go over the batch
    for i in range(bs):
        # Note, ivf-pq does not guarantee the ordered input, hence the use of intersect1d
        n += cp.intersect1d(found_indices[i, :k], ground_truth[i, :k]).size
        # To-do: Change to account for equidistant indices that are not captured.
    
    #recall = n / found_indices.size
    recall = n / (bs * ground_truth.shape[1])
    return recall

def load_all_input_configs(config_path, logger):
    """Load and process all input configuration files"""
    filter_config = load_config(os.path.join(config_path, 'filter_config.yaml'))
    params_config = load_config(os.path.join(config_path, 'params_config.yaml'))
    
    # Process configurations
    active_filters = filter_config.get('active_filters', ['low_rated', 'high_rated', 'mid_rated'])
    filter_configurations = [(filter_config['filter_configurations'][name]['ranges'], name) 
                           for name in active_filters if name in filter_config['filter_configurations']]
    
    config = {
        'filter_configurations': filter_configurations,
        'global_filter_settings': filter_config.get('global_filter_settings', {}),
        'rating_distribution': filter_config.get('rating_distribution', {}),  # Add this line
        'cagra_params': params_config['cagra_params'],
        'hnsw_params': params_config['hnsw_params'],
        'data_sizes': params_config.get('data_sizes', [5010000]),
        **params_config.get('global_settings', {}),
        **params_config.get('results_settings', {})
    }
    
    # Set defaults
    defaults = {'num_queries': 10000, 'batch_size': 100, 'num_workers_params': 1, 
                'num_workers_throughput': 2, 'persistent': False, 'run_time_seconds': 30,
                'results_dir': 'comprehensive_comparison_results-test', 'save_intermediate': True, 
                'cleanup_pause_seconds': 30}
    
    for key, default in defaults.items():
        config.setdefault(key, default)
    
    logger.info(f"Loaded {len(config['filter_configurations'])} filters, CAGRA params, HNSW params")
    return config

def load_config(config_path, default_config=None):
    """Load configuration from YAML file with fallback to default."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        if default_config:
            print(f"Config file {config_path} not found, using default configuration")
            return default_config
        else:
            raise
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        if default_config:
            return default_config
        else:
            raise

def setup_progress_tracking(log_file="cagra_search.log"):
    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Create logger
    progress_logger = logging.getLogger('progress_tracker')
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False  # Don't propagate to root logger
    
    # Console handler - for real-time viewing
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('🔍 PROGRESS: %(message)s'))
    progress_logger.addHandler(console_handler)
    
    # File handler - for persistent logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    progress_logger.addHandler(file_handler)
    
    return progress_logger

import cupy as cp
import numpy as np
from cuvs.neighbors import filters
import time
import matplotlib.pyplot as plt

def count_selected_samples(bitset):
    """Count bits set to 1 in a bitset using an efficient bit counting algorithm."""
    bitset_cpu = cp.asnumpy(bitset)
    # Convert each uint32 to binary and count through bit manipulation
    count = 0
    for val in bitset_cpu:
        # Brian Kernighan's algorithm
        bits = np.uint32(val)
        bit_count = 0
        while bits:
            bits &= bits - 1
            bit_count += 1
        count += bit_count
    
    return count

def create_rating_distribution_bitquery(
    n_samples, 
    selected_ranges=None, 
    device_id=0, 
    verbose=False,
    batch_size=5_000_000,
    config=None  # Default batch size appropriate for large datasets
):
    """
    Create a single query-specific filter based on rating distribution.
    Each bit in the bitset corresponds to a vector in the dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of vectors in the dataset
    selected_ranges : list or None
        Rating ranges to include in filter
    device_id : int
        GPU device to use
    verbose : bool
        Whether to print progress information
    batch_size : int
        Number of positions to process in each batch (for memory efficiency)
    """
    device_id = _validate_device_id(device_id)
    with cp.cuda.Device(device_id):
        # Rating distribution data
        #TO-do: make this a config file
        rating_distribution = config['rating_distribution']

        total = sum(rating_distribution.values())
        rating_probabilities = {k: v/total for k, v in rating_distribution.items()}

        # Default to all ranges if none specified
        if selected_ranges is None:
            selected_ranges = list(rating_distribution.keys())

        # Calculate target percentage based on selected ranges
        ranges = list(rating_probabilities.keys())
        selected_indices = [i for i, r in enumerate(ranges) if r in selected_ranges]
        selected_prob_total = sum(rating_probabilities[ranges[i]] for i in selected_indices)
        
        # Calculate number of bits to set
        target_bits = int(n_samples * selected_prob_total)
        
        if verbose:
            print(f"Target percentage: {selected_prob_total*100:.2f}%")
            print(f"Target bits to set: {target_bits}")
        
        # Initialize bitset
        n_bitmap = int(cp.ceil(n_samples / 32))
        bitset = cp.zeros(n_bitmap, dtype=cp.uint32)
        
        # Generate random positions
        all_positions = cp.arange(n_samples)
        cp.random.shuffle(all_positions)
        selected_positions = all_positions[:target_bits]
        
        # Sort positions for better memory access patterns
        selected_positions = cp.sort(selected_positions)
        
        # Define kernel that works directly with positions
        set_bit_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void set_bit(unsigned int *bitmap, const long long *positions, int n) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < n) {
                long long pos = positions[tid];
                unsigned int word_idx = pos >> 5;  // Divide by 32
                unsigned int bit_idx = pos & 31;   // Modulo 32
                atomicOr(&bitmap[word_idx], 1u << bit_idx);
            }
        }
        ''', 'set_bit')
        
        # Process in batches for memory efficiency
        threads_per_block = 256
        for batch_start in range(0, len(selected_positions), batch_size):
            batch_end = min(batch_start + batch_size, len(selected_positions))
            batch_positions = selected_positions[batch_start:batch_end]
            
            # Set kernel configuration for this batch
            blocks_per_grid = (len(batch_positions) + threads_per_block - 1) // threads_per_block
            
            # Launch kernel for this batch
            set_bit_kernel((blocks_per_grid,), (threads_per_block,),
                         (bitset, batch_positions, len(batch_positions)))
            
            if verbose and (batch_start == 0 or batch_end == len(selected_positions)):
                print(f"Processed {batch_end}/{len(selected_positions)} positions")
        
        # Print info if verbose is enabled
        if verbose:
            num_bits = count_selected_samples(bitset)
            print(f"Created bitquery with {num_bits} of {n_samples} points selected ({num_bits/n_samples*100:.2f}%)")
        
        return bitset

def _validate_device_id(device_id):
    """Validate the device_id and return a valid ID or raise an error"""
    device_count = cp.cuda.runtime.getDeviceCount()
    
    if device_id < 0 or device_id >= device_count:
        # If invalid, default to device 0 if available
        if device_count > 0:
            print(f"⚠️ Warning: device_id={device_id} is invalid. Using device 0 instead.")
            return 0
        else:
            raise ValueError(f"No CUDA devices available")
    
    return device_id

def create_rating_filter(n_samples, n_queries, valid_ranges, device_id=0, max_memory_gb=None, verbose=True, visualize=False, config=None):
    """
    Create a filter based on selected rating ranges
    
    Parameters:
    -----------
    n_samples : int
        Number of reference points
    n_queries : int
        Number of query points
    selected_ranges : list of str
        List of rating ranges to include
    device_id : int
        GPU device ID to use for computations
    max_memory_gb : float, optional
        Maximum GPU memory to use in gigabytes
    verbose : bool
        Whether to print progress information
    visualize : bool
        Whether to create and show distribution visualizations
    """
    start_time = time.time()
    
    # Validate device_id
    device_id = _validate_device_id(device_id)
    
    print("valid ranges", valid_ranges)
    bitquery = create_rating_distribution_bitquery(
        n_samples, valid_ranges, 
        device_id=device_id,
        verbose=verbose,
        config=config
    )
    
    filter_obj = filters.from_bitset(bitquery)
    
    if verbose:
        total_time = time.time() - start_time
        print(f"Filter creation time: {total_time:.3f} seconds")
    
    return filter_obj, bitquery

def load_quantized_vectors(quantization_folder, max_index, quantization_type, logger=None):
    """
    Load all quantized vectors up to max_index from files.
    
    Args:
        quantization_folder: Base folder containing quantized vector files
        max_index: Maximum index to load up to
        quantization_type: One of 'full_precision', 'scalar', or 'binary'
        
    Returns:
        Numpy array of all loaded vectors up to max_index
    """
    import os
    import numpy as np
    
    suffix_map = {
    'full_precision': '_full.npy',
    'scalar': '_scalar.npy',
    'binary': '_binary.npy',
    'fp': '_full.npy',
    'sq': '_scalar.npy',
    'bq': '_binary.npy',
    'fp16': '.npy',
    'half_precision': '_fp16.npy'
    }
    
    # Use the quantization_folder directly as the full path
    folder_path = quantization_folder
    suffix = suffix_map[quantization_type]
    
    logger.debug(f"Loading {quantization_type} vectors up to index {max_index}")
    logger.debug(f"Looking in folder: {folder_path}")
    
    logger.debug(f"Loading {quantization_type} vectors up to index {max_index}")
    
    # Get sorted list of files in the folder
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(suffix)])
    
    if not all_files:
        raise ValueError(f"No files with suffix {suffix} found in {folder_path}")
    
    # Load files sequentially until we have enough data
    vectors_list = []
    total_vectors = 0
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        
        # Load the vectors from this file
        vectors = np.load(file_path)
        
        # Check and convert dtype if necessary
        if vectors.dtype == np.float64:
            logger.debug(f"Converting {file} from float64 to float32")
            vectors = vectors.astype(np.float32)
        elif vectors.dtype in [np.float32, np.int8]:
            logger.debug(f"Vectors in {file} already in efficient format: {vectors.dtype}")
        else:
            logger.debug(f"Note: Vectors in {file} using dtype: {vectors.dtype}")
        
        vectors_list.append(vectors)
        total_vectors += vectors.shape[0]
        
        logger.debug(f"Loaded {file}, total vectors now: {total_vectors}")
        
        # Stop if we've loaded enough data
        if total_vectors > max_index:
            break
    
    # Stack all loaded vectors
    all_vectors = np.vstack(vectors_list)
    logger.debug(f"Total {quantization_type} vectors loaded: {all_vectors.shape[0]}")
    
    # Ensure we have enough vectors
    if max_index >= all_vectors.shape[0]:
        raise ValueError(f"Requested max_index {max_index} exceeds available vectors ({all_vectors.shape[0]})")
    
    # Return all vectors up to the requested count
    return all_vectors

# New function to determine appropriate metric and build algorithm
def get_algorithm_settings(quantization_type):
    """
    Get appropriate metric and build algorithm based on quantization type.
    
    Args:
        quantization_type: Type of quantization
        
    Returns:
        tuple: (metric, build_algo)
    """
    if quantization_type in ["binary", "bq"]:
        return "bitwise_hamming", "iterative_cagra_search"
    elif quantization_type in ["scalar", "sq"]:
        return "sqeuclidean", "nn_descent"
    else:  # full_precision or fp
        return "sqeuclidean", "nn_descent"

def load_vector_data(vectors=None, queries=None, vectors_fp=None, queries_fp=None, 
                     quantization_type="fp", quantization_folder=None,
                     train_indices=None, val_indices=None, 
                     refinement=False, rank=False, logger=None):
    """
    Load vector data by first loading all vectors up to max index,
    then separating into queries and reference vectors based on indices.
    
    Args:
        vectors: Pre-quantized vectors (used if quantization_folder is None)
        queries: Pre-quantized queries (used if quantization_folder is None)
        vectors_fp: Full precision vectors (for ground truth/refinement/ranking)
        queries_fp: Full precision queries (for ground truth/refinement/ranking)
        quantization_type: Type of quantization ('fp', 'sq', 'bq' or 'full_precision', 'scalar', 'binary')
        quantization_folder: Folder containing pre-quantized vectors (if None, uses provided vectors)
        train_indices: Indices for training vectors to load (required if quantization_folder is provided)
        val_indices: Indices for validation vectors to load (required if quantization_folder is provided)
        refinement: Whether refinement will be used (affects which vectors need to be loaded)
        rank: Whether ranking will be used (affects which vectors need to be loaded)
        
    Returns:
        tuple: (vectors, queries, vectors_fp, queries_fp)
    """
    import numpy as np
    
    if quantization_folder is not None:
        logger.debug(f"Loading pre-quantized vectors from {quantization_folder}")
        
        if train_indices is None or val_indices is None:
            raise ValueError("train_indices and val_indices must be provided when using quantization_folder")
        
        # Find maximum required index
        max_index = max(np.max(train_indices), np.max(val_indices))
        print("max_index ",max_index)
        # CAGRA: For fp16/half_precision, load from half_precision for both main and ground-truth vectors
        if quantization_type in ["fp16", "half_precision"]:
            all_vectors_fp16 = load_quantized_vectors(quantization_folder, max_index, 'fp16', logger)
            vectors = all_vectors_fp16[train_indices]
            queries = all_vectors_fp16[val_indices]
            vectors_fp = vectors
            queries_fp = queries
        elif quantization_type in ["full_precision", "fp"]:
            all_vectors_fp = load_quantized_vectors(quantization_folder, max_index, 'full_precision', logger)
            vectors = all_vectors_fp[train_indices]
            queries = all_vectors_fp[val_indices]
            vectors_fp = vectors
            queries_fp = queries
        else:
            # For quantized vectors, load both quantized and full precision
            all_vectors_q = load_quantized_vectors(quantization_folder, max_index, quantization_type, logger)
            vectors = all_vectors_q[train_indices]
            queries = all_vectors_q[val_indices]
            # Always load full precision for ground truth calculation
            if refinement or rank or True:  # Always true to force loading
                all_vectors_fp = load_quantized_vectors(quantization_folder, max_index, 'full_precision', logger)
                vectors_fp = all_vectors_fp[train_indices]
                queries_fp = all_vectors_fp[val_indices]
    else:
        # Using provided vectors (on-the-fly or pre-quantized)
        if vectors is None or queries is None:
            raise ValueError("vectors and queries must be provided when not using quantization_folder")
            
        # For full precision, ensure vectors_fp and queries_fp are set
        if quantization_type in ["full_precision", "fp", "fp16", "half_precision"]:
            if vectors_fp is None:
                vectors_fp = vectors
            if queries_fp is None:
                queries_fp = queries
                
        # For binary or scalar, ensure full precision is available for ground truth
        elif vectors_fp is None or queries_fp is None:
            raise ValueError("vectors_fp and queries_fp must be provided for ground truth calculation")
    
    logger.debug(f"Vectors shape: {vectors.shape}")
    logger.debug(f"Queries shape: {queries.shape}")
    if vectors_fp is not None and queries_fp is not None:
        logger.debug(f"Full precision vectors shape: {vectors_fp.shape}")
        logger.debug(f"Full precision queries shape: {queries_fp.shape}")
    
    return vectors, queries, vectors_fp, queries_fp

def generate_one_time_ground_truth(train_vectors, val_vectors, k=10, filter_obj=None, logger=None):
    """
    Generate ground truth using provided train and validation vectors.
    No explicit GPU memory pool is used.
    
    Args:
        train_vectors: Reference vectors for search
        val_vectors: Query vectors
        k: Number of nearest neighbors to find
        filter_obj: Optional filter object to apply during search
    """

    logger.debug(f"Generating ground truth using {val_vectors.shape[0]} validation vectors")
    if filter_obj is not None:
        logger.debug("Using filter for ground truth calculation")

    # Import cupy if available, otherwise use numpy
    xp = import_with_fallback("cupy", "numpy")

    logger.debug(f"Train vectors shape: {train_vectors.shape}")
    logger.debug(f"Validation vectors shape: {val_vectors.shape}")

    # Generate ground truth directly using provided vectors
    logger.debug("Calculating ground truth...")
    try:
        # Pass the filter to calc_truth if provided
        _, gt_indices = calc_truth(train_vectors, val_vectors, k, metric="sqeuclidean", filter=filter_obj)

        # Ensure gt_indices is numpy array and correct type
        if hasattr(gt_indices, 'get'):  # For cupy arrays
            gt_indices = gt_indices.get()
        gt_indices = np.asarray(gt_indices, dtype=np.int32)

        logger.debug(f"Ground truth generated with shape: {gt_indices.shape}")

        # Release all GPU memory (if using cupy)
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        return gt_indices

    except Exception as e:
        logger.error(f"Error generating ground truth: {e}")
        # Also try to free memory on error
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        raise


def calculate_recall_with_batching(queries, cagra_index, search_params, filter_obj, k, batch_size=32):
    n_queries = queries.shape[0]
    all_indices = np.zeros((n_queries, k), dtype=np.int32)
    total_search_time = 0
    
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batch_queries = queries[start_idx:end_idx]
        batch_queries_gpu = cp.asarray(batch_queries)
        
        # Time the batch
        t1 = time.time()
        
        if filter_obj is not None:
            distances, indices = cagra.search(search_params, cagra_index, batch_queries_gpu, k=k, filter=filter_obj)
        else:
            distances, indices = cagra.search(search_params, cagra_index, batch_queries_gpu, k=k)
        
        search_time = time.time() - t1
        total_search_time += search_time
        
        # Store results
        indices_np = cp.asnumpy(indices)
        all_indices[start_idx:end_idx] = indices_np
    
    return all_indices, total_search_time

def build_hnsw_index(params, vectors):
    """Build a FAISS HNSW index and return it."""
    d = vectors.shape[1]  # Vector dimension

    # Extract parameters
    M = params.get('M', 16)
    efConstruction = params.get('efConstruction', 40)
    efSearch = params.get('efSearch', 16)
    quantization_type = params.get('quantization_type', 'fp')

    # Select index type based on quantization
    if quantization_type == 'sq':
        # Scalar quantized HNSW
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
    elif quantization_type == 'bq':
        # Binary quantized HNSW (approximation using binary flat)
        nbits = (d + 7) // 8 * 8  # Round up to nearest multiple of 8
        index = faiss.IndexBinaryHNSW(nbits, M)
        # Convert to binary
        vectors = (vectors > 0).astype('uint8')
        vectors = np.packbits(vectors, axis=1)
    elif quantization_type in ['fp16', 'half_precision']:
        print("here building index")
        # FP16 scalar quantized HNSW using index_factory
        index = faiss.index_factory(d, f"HNSW{M}_SQfp16")
        # Set HNSW construction/search params if available
        if hasattr(index, 'hnsw'):
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = efSearch
        start_time = time.time()
        # Train and add using full-precision vectors (FAISS will quantize internally)
        if hasattr(index, 'train'):
            index.train(vectors)
        index.add(vectors)
        build_time = time.time() - start_time
        return index, build_time
    else:
        # Full precision HNSW
        index = faiss.IndexHNSWFlat(d, M)

    # Set HNSW construction parameters
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    
    start_time = time.time()

    # Train and add vectors
    if hasattr(index, 'train'):
        index.train(vectors)
    index.add(vectors)

    build_time = time.time() - start_time
    return index, build_time

import argparse
import importlib
import os
import sys
import warnings
import pylibraft

#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import numpy as np


def dtype_from_filename(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".fbin":
        return np.float32
    if ext == ".hbin":
        return np.float16
    elif ext == ".ibin":
        return np.int32
    elif ext == ".u8bin":
        return np.ubyte
    elif ext == ".i8bin":
        return np.byte
    else:
        raise RuntimeError("Not supported file extension" + ext)


def suffix_from_dtype(dtype):
    if dtype == np.float32:
        return ".fbin"
    if dtype == np.float16:
        return ".hbin"
    elif dtype == np.int32:
        return ".ibin"
    elif dtype == np.ubyte:
        return ".u8bin"
    elif dtype == np.byte:
        return ".i8bin"
    else:
        raise RuntimeError("Not supported dtype extension" + dtype)


def memmap_bin_file(
    bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32
):
    extent_itemsize = np.dtype(size_dtype).itemsize
    offset = int(extent_itemsize) * 2
    if bin_file is None:
        return None
    if dtype is None:
        dtype = dtype_from_filename(bin_file)

    if mode[0] == "r":
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        if shape is None:
            shape = (a[0], a[1])
        else:
            shape = tuple(
                [
                    aval if sval is None else sval
                    for aval, sval in zip(a, shape)
                ]
            )

        return np.memmap(
            bin_file, mode=mode, dtype=dtype, offset=offset, shape=shape
        )
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")

        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        a[0] = shape[0]
        a[1] = shape[1]
        a.flush()
        del a
        fp = np.memmap(
            bin_file, mode="r+", dtype=dtype, offset=offset, shape=shape
        )
        return fp


def write_bin(fname, data):
    print("writing", fname, data.shape, data.dtype, "...")
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


def import_with_fallback(primary_lib, secondary_lib=None, alias=None):
    """
    Attempt to import a primary library, with an optional fallback to a
    secondary library.
    Optionally assigns the imported module to a global alias.

    Parameters
    ----------
    primary_lib : str
        Name of the primary library to import.
    secondary_lib : str, optional
        Name of the secondary library to use as a fallback. If `None`,
        no fallback is attempted.
    alias : str, optional
        Alias to assign the imported module globally.

    Returns
    -------
    module or None
        The imported module if successful; otherwise, `None`.

    Examples
    --------
    >>> xp = import_with_fallback('cupy', 'numpy')
    >>> mod = import_with_fallback('nonexistent_lib')
    >>> if mod is None:
    ...     print("Library not found.")
    """
    try:
        module = importlib.import_module(primary_lib)
    except ImportError:
        if secondary_lib is not None:
            try:
                module = importlib.import_module(secondary_lib)
            except ImportError:
                module = None
        else:
            module = None
    if alias and module is not None:
        globals()[alias] = module
    return module


xp = import_with_fallback("cupy", "numpy")
rmm = import_with_fallback("rmm")
gpu_system = False


def force_fallback_to_numpy():
    global xp, gpu_system
    xp = import_with_fallback("numpy")
    gpu_system = False
    warnings.warn(
        "Consider using a GPU-based system to greatly accelerate "
        " generating groundtruths using cuVS."
    )


if rmm is not None:
    gpu_system = True
    try:
        from pylibraft.common import DeviceResources
        from rmm.allocators.cupy import rmm_cupy_allocator

        from cuvs.neighbors.brute_force import build, search
    except ImportError:
        # RMM is available, cupy is available, but cuVS is not
        force_fallback_to_numpy()
else:
    # No RMM, no cuVS, but cupy is available
    force_fallback_to_numpy()


def generate_random_queries(n_queries, n_features, dtype=xp.float32):
    print("Generating random queries")
    if xp.issubdtype(dtype, xp.integer):
        queries = xp.random.randint(
            0, 255, size=(n_queries, n_features), dtype=dtype
        )
    else:
        queries = xp.random.uniform(size=(n_queries, n_features)).astype(dtype)
    return queries


def choose_random_queries(dataset, n_queries):
    print("Choosing random vector from dataset as query vectors")
    query_idx = xp.random.choice(
        dataset.shape[0], size=(n_queries,), replace=False
    )
    return dataset[query_idx, :]


def cpu_search(dataset, queries, k, metric="squeclidean"):
    """
    Find the k nearest neighbors for each query point in the dataset using the
    specified metric.

    Parameters
    ----------
    dataset : numpy.ndarray
        An array of shape (n_samples, n_features) representing the dataset.
    queries : numpy.ndarray
        An array of shape (n_queries, n_features) representing the query
        points.
    k : int
        The number of nearest neighbors to find.
    metric : str, optional
        The distance metric to use. Can be 'squeclidean' or 'inner_product'.
        Default is 'squeclidean'.

    Returns
    -------
    distances : numpy.ndarray
        An array of shape (n_queries, k) containing the distances
        (for 'squeclidean') or similarities
        (for 'inner_product') to the k nearest neighbors for each query.
    indices : numpy.ndarray
        An array of shape (n_queries, k) containing the indices of the
        k nearest neighbors in the dataset for each query.

    """
    if metric == "squeclidean":
        diff = queries[:, xp.newaxis, :] - dataset[xp.newaxis, :, :]
        dist_sq = xp.sum(diff**2, axis=2)  # Shape: (n_queries, n_samples)

        indices = xp.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
        distances = xp.take_along_axis(dist_sq, indices, axis=1)

        sorted_idx = xp.argsort(distances, axis=1)
        distances = xp.take_along_axis(distances, sorted_idx, axis=1)
        indices = xp.take_along_axis(indices, sorted_idx, axis=1)

    elif metric == "inner_product":
        similarities = xp.dot(
            queries, dataset.T
        )  # Shape: (n_queries, n_samples)

        neg_similarities = -similarities
        indices = xp.argpartition(neg_similarities, kth=k - 1, axis=1)[:, :k]
        distances = xp.take_along_axis(similarities, indices, axis=1)

        sorted_idx = xp.argsort(-distances, axis=1)

    else:
        raise ValueError(
            "Unsupported metric in cuvs-bench-cpu. "
            "Use 'squeclidean' or 'inner_product' or use the GPU package"
            "to use any distance supported by cuVS."
        )

    distances = xp.take_along_axis(distances, sorted_idx, axis=1)
    indices = xp.take_along_axis(indices, sorted_idx, axis=1)

    return distances, indices

def calc_truth(dataset, queries, k, metric="sqeuclidean", filter=None):
    """
    Calculate ground truth nearest neighbors with optional filtering.
    
    Parameters:
    -----------
    dataset : array-like
        Reference vectors
    queries : array-like
        Query vectors
    k : int
        Number of nearest neighbors to find
    metric : str
        Distance metric to use
    filter : object, optional
        Filter object to apply during search
    
    Returns:
    --------
    tuple: (distances, indices)
    """
    queries = xp.asarray(queries, dtype=xp.float32)
    dataset = xp.asarray(dataset, dtype=xp.float32)
    
    print("Building index for full dataset ({} vectors)...".format(dataset.shape[0]))
    
    if gpu_system:
        resources = DeviceResources()
        
        try:
            # Build index with full dataset
            index = build(dataset, metric=metric)
            
            # Search with optional filter
            print("Searching with full dataset...")
            if filter is not None:
                D, Ind = search(index, queries, k, prefilter=filter)
            else:
                D, Ind = search(index, queries, k)
                
            resources.sync()
            
            # Convert results back to CPU before returning
            distances = xp.asnumpy(D)
            indices = xp.asnumpy(Ind)
            
        finally:
            # Clean up GPU memory
            if 'index' in locals():
                del index
            del dataset, queries
            mem_pool = xp.get_default_memory_pool()
            mem_pool.free_all_blocks()
            
    else:
        # CPU search doesn't support filters
        if filter is not None:
            print("Warning: Filters not supported in CPU implementation")
        distances, indices = cpu_search(dataset, queries, k, metric=metric)

    return distances, indices

# Add memory monitoring imports
import threading
import time
from collections import defaultdict

class MemoryMonitor:
    """
    A class to monitor VRAM and RAM usage in real-time during benchmark execution.
    """
    
    def __init__(self, logger=None, monitor_interval=0.1):
        """
        Initialize the memory monitor.
        
        Args:
            logger: Logger instance for reporting
            monitor_interval: How often to check memory usage (in seconds)
        """
        self.logger = logger
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Memory usage tracking
        self.max_vram_used = 0
        self.max_ram_used = 0
        self.vram_usage_history = []
        self.ram_usage_history = []
        
        # Initialize pynvml for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0
            total_vram = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total
            self.total_vram_gb = total_vram / (1024**3)
        except Exception as e:
            self.gpu_available = False
            self.total_vram_gb = 0
            if logger:
                logger.warning(f"GPU monitoring not available: {e}")
        
        # Get total system RAM
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if logger:
            logger.info(f"Memory Monitor initialized - Total VRAM: {self.total_vram_gb:.1f}GB, Total RAM: {self.total_ram_gb:.1f}GB")
    
    def get_current_vram_usage(self):
        """Get current VRAM usage in GB."""
        if not self.gpu_available:
            return 0
        
        try:
            # Try cupy first (if available)
            import cupy as cp
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            return used_mem / (1024**3)
        except:
            try:
                # Fallback to pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                return mem_info.used / (1024**3)
            except:
                return 0
    
    def get_current_ram_usage(self):
        """Get current RAM usage in GB."""
        try:
            return psutil.virtual_memory().used / (1024**3)
        except:
            return 0
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            # Get current usage
            vram_used = self.get_current_vram_usage()
            ram_used = self.get_current_ram_usage()
            
            # Update maximums
            self.max_vram_used = max(self.max_vram_used, vram_used)
            self.max_ram_used = max(self.max_ram_used, ram_used)
            
            # Store history
            timestamp = time.time()
            self.vram_usage_history.append((timestamp, vram_used))
            self.ram_usage_history.append((timestamp, ram_used))
            
            # Keep only last 1000 measurements to prevent memory bloat
            if len(self.vram_usage_history) > 1000:
                self.vram_usage_history = self.vram_usage_history[-1000:]
                self.ram_usage_history = self.ram_usage_history[-1000:]
            
            time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.logger:
            self.logger.info("Started memory monitoring")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if self.logger:
            self.logger.info("Stopped memory monitoring")
    
    def get_peak_usage(self):
        """Get peak memory usage statistics."""
        vram_percent = (self.max_vram_used / self.total_vram_gb * 100) if self.total_vram_gb > 0 else 0
        ram_percent = (self.max_ram_used / self.total_ram_gb * 100) if self.total_ram_gb > 0 else 0
        
        return {
            'max_vram_gb': self.max_vram_used,
            'max_ram_gb': self.max_ram_used,
            'max_vram_percent': vram_percent,
            'max_ram_percent': ram_percent,
            'total_vram_gb': self.total_vram_gb,
            'total_ram_gb': self.total_ram_gb
        }
    
    def log_current_usage(self, context=""):
        """Log current memory usage."""
        if not self.logger:
            return
        
        vram_used = self.get_current_vram_usage()
        ram_used = self.get_current_ram_usage()
        
        vram_percent = (vram_used / self.total_vram_gb * 100) if self.total_vram_gb > 0 else 0
        ram_percent = (ram_used / self.total_ram_gb * 100) if self.total_ram_gb > 0 else 0
        
        self.logger.info(
            f"Memory usage{' (' + context + ')' if context else ''}: "
            f"VRAM: {vram_used:.1f}GB ({vram_percent:.1f}%), "
            f"RAM: {ram_used:.1f}GB ({ram_percent:.1f}%)"
        )
    
    def log_peak_usage(self, context=""):
        """Log peak memory usage since monitoring started."""
        if not self.logger:
            return
        
        stats = self.get_peak_usage()
        
        self.logger.info(
            f"Peak memory usage{' (' + context + ')' if context else ''}: "
            f"VRAM: {stats['max_vram_gb']:.1f}GB ({stats['max_vram_percent']:.1f}%), "
            f"RAM: {stats['max_ram_gb']:.1f}GB ({stats['max_ram_percent']:.1f}%)"
        )
    
    def reset_peak_tracking(self):
        """Reset peak usage tracking."""
        self.max_vram_used = 0
        self.max_ram_used = 0
        self.vram_usage_history.clear()
        self.ram_usage_history.clear()
        
        if self.logger:
            self.logger.info("Reset peak memory usage tracking")
    
    def save_usage_plot(self, filename="memory_usage.png", title="Memory Usage Over Time"):
        """Save a plot of memory usage over time."""
        if not self.vram_usage_history and not self.ram_usage_history:
            if self.logger:
                self.logger.warning("No memory usage history to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Convert timestamps to datetime objects
            if self.vram_usage_history:
                vram_times = [datetime.fromtimestamp(t) for t, _ in self.vram_usage_history]
                vram_values = [v for _, v in self.vram_usage_history]
                
                ax1.plot(vram_times, vram_values, label='VRAM Usage', color='red', linewidth=2)
                ax1.axhline(y=self.total_vram_gb, color='red', linestyle='--', alpha=0.5, label=f'Total VRAM ({self.total_vram_gb:.1f}GB)')
                ax1.set_ylabel('VRAM Usage (GB)')
                ax1.set_title('GPU Memory Usage')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            if self.ram_usage_history:
                ram_times = [datetime.fromtimestamp(t) for t, _ in self.ram_usage_history]
                ram_values = [v for _, v in self.ram_usage_history]
                
                ax2.plot(ram_times, ram_values, label='RAM Usage', color='blue', linewidth=2)
                ax2.axhline(y=self.total_ram_gb, color='blue', linestyle='--', alpha=0.5, label=f'Total RAM ({self.total_ram_gb:.1f}GB)')
                ax2.set_ylabel('RAM Usage (GB)')
                ax2.set_title('System Memory Usage')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
            plt.xticks(rotation=45)
            
            plt.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"Memory usage plot saved to {filename}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save memory usage plot: {e}")

def create_memory_monitor(logger=None, monitor_interval=0.1):
    """
    Create and return a memory monitor instance.
    
    Args:
        logger: Logger instance
        monitor_interval: How often to check memory usage (in seconds)
        
    Returns:
        MemoryMonitor instance
    """
    return MemoryMonitor(logger=logger, monitor_interval=monitor_interval)
