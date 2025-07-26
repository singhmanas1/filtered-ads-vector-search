# Benchmarking CAGRA and HNSW on Amazon Reviews Dataset

## Overview

This project benchmarks two vector search algorithms - CAGRA (GPU-accelerated) and HNSW (CPU-based) - using Amazon product reviews dataset. The benchmarking pipeline includes data preprocessing, embedding generation using NVIDIA's LLaMA model, and performance evaluation of different search algorithms with configurable filtering mechanisms.

## Features

- Amazon Reviews dataset download and preprocessing
- Embedding generation using NVIDIA LLaMA embedding model
- Support for both FP32 and FP16 precision embeddings
- CAGRA and HNSW algorithm benchmarking
- Configurable filtering system for filtered search evaluation

## Prerequisites

- Python 3.8+
- NVIDIA GPU (for CAGRA benchmarking)
- NVIDIA NGC API key (for embedding model)
- Required Python packages (see requirements_umap.txt)

## 1. Setup

### 1.1 Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 1.2 Setup Embedding Server

1. Get your API key from [ngc.nvidia.com](https://ngc.nvidia.com)
2. Add the API key to the `embedder_run.sh` file
3. Launch the NVIDIA LLaMA embedding model server:

```bash
chmod +x embedder_run.sh
./embedder_run.sh
```

The server will be available on ports 8000 and 8001. This NIMs server is used for creating embeddings of data.

## 2. Data Processing

### 2.1 Download Amazon Reviews Dataset

Run the download script to fetch category-wise review data:

```bash
python download_amazon_dataset.py
```

This creates an `amazon_reviews_2023_categories` directory with JSONL files for each product category.

### 2.2 Generate Embeddings

The embedding generation supports both FP32 and FP16 precision:

#### FP32 Embeddings (Default)
```bash
python exec_embed.py \
    --input_dir amazon_reviews_2023_categories/raw/review_categories \
    --output_dir ./amazon_embeddings_categories
```

#### FP16 Embeddings (Half Precision)
```bash
python exec_embed.py \
    --input_dir amazon_reviews_2023_categories/raw/review_categories \
    --output_dir ./amazon_embeddings_categories \
    --fp16
```

**Note:** `exec_embed.py` uses the LLaMA embedding model to generate 64-dimensional embeddings from the review text data.
5M dataset embeddings for reviews dataset have been uploaded [here](https://us-east-1.console.aws.amazon.com/s3/buckets/amazon-ads-vector-search?region=us-east-1&bucketType=general&prefix=amazon_reviews_embeddings/&showversions=false)

## 3. Benchmarking

### 3.1 Configuration Files

Before running benchmarks, ensure you have the required configuration files:

#### `filter_config.yaml`
This file specifies which filters to use during benchmarking. It defines different filter types and their parameters:
- Filter names (e.g., "mid-rated", "high-rated", "low-rated")
- Filter selectivity percentages
- Filter application strategies

#### `cagra_config.yaml`
Contains CAGRA algorithm parameters and settings:
- Graph construction parameters for different data_sizes.The `data_sizes: [5010000]` specifies the size of your dataset.
- Search parameters - Like number of Queries to search for
- Memory usage settings
- Optimization flags

### 3.2 Run Benchmarks

#### CAGRA Benchmarking
```bash
python cuvs_bench_grid_search_v2.py \
    --quantization-folder ./ \
    --quantization-folder-path /home/ubuntu/amazon-ads-search-slikhite-test/amazon-embeddings \
    --algo-type cagra
```

For background process:
```bash
nohup python cuvs_bench_grid_search_v2.py \
    --quantization-folder ./ \
    --quantization-folder-path /home/ubuntu/amazon-ads-search-slikhite-test/amazon-embeddings \
    --algo-type cagra > benchmark-cagra.log 2>&1 &
```

#### HNSW Benchmarking
```bash
python cuvs_bench_grid_search_v2.py \
    --quantization-folder ./ \
    --quantization-folder-path /home/ubuntu/amazon-ads-search-slikhite-test/amazon-embeddings  \
    --algo-type hnsw
```

For background process:
```bash
nohup python cuvs_bench_grid_search_v2.py \
    --quantization-folder ./ \
    --quantization-folder-path /home/ubuntu/amazon-ads-search-slikhite-test/amazon-embeddings \
    --algo-type hnsw > benchmark-hnsw.log 2>&1 &
```

Note: HNSW Index building and Search parameters are not specified in the cagra_config.yaml file at the moment.

## 4. Results

Benchmark results comparing **CAGRA** and **HNSW** under different filter conditions.
Metrics reported: **Best Recall** and **Best QPS (Queries Per Second)**.


| Filter      | Filtering % | HNSW Recall | HNSW QPS   | CAGRA Recall | CAGRA QPS   |
|-------------|-------------|-------------|------------|---------------|-------------|
| Low Rated   | 83.63%      | 0.91401     | 21,890.05  | **0.92286**   | **46,513.16** |
| High Rated  | 15.61%      | 0.91867     | 22,672.58  | **0.93870**   | **35,638.04** |
| Mid Rated   | 0.77%       | 0.80142     | 24,271.58  | **0.97251**   | **8,439.41**  |

## 5. Misc

### 5.1 How Filtering Works

The filtering system is designed to evaluate search performance under different data selectivity conditions. Here's how it works:

#### Filter Object Implementation
- **Unified Filter Object**: Both CAGRA and HNSW use the same filter object for consistency
- **Random Bit Setting**: The filter object randomly sets bits in a binary mask based on the specified filter type
- **Filter Configuration**: `filter_config.yaml` specifies which filters to apply during benchmarking

#### Filter Process Example
Let's say you're using a "mid-rated" filter:

1. **Filter Selection**: The system reads `filter_config.yaml` and identifies the "mid-rated" filter configuration
2. **Data Filtering**: This filter is configured to filter out 99% of the data (keeping only 1% for search)
3. **Bit Vector Creation**: A bit query vector of size `num_vectors` is created
4. **Random Bit Assignment**: Only 0.77% of the data is randomly set to `1` (searchable), while the remaining 99.23% is set to `0` (filtered out)
5. **Search Application**: During search operations, the filter object is applied to restrict the search space to only the vectors marked with `1`

#### Filter Types
The filtering system supports various filter types that simulate different real-world scenarios:
- **High selectivity filters**: Filter out 95-99% of data (e.g., "premium products")
- **Medium selectivity filters**: Filter out 70-90% of data (e.g., "mid-rated products")
- **Low selectivity filters**: Filter out 10-50% of data (e.g., "recent reviews")

#### Benefits of Filtered Search
- **Real-world simulation**: Mimics scenarios where you need to search within specific subsets of data
- **Performance evaluation**: Measures how algorithms perform under different data selectivity conditions
- **Consistency**: Same filtering logic applied to both CAGRA and HNSW for fair comparison

### Known issues:
1. HNSW parameters are not specified in the yaml file as of now. They are in this function: def generate_hnsw_parameter_grid in [./cuvs_bench_grid_search_v2.py](./cuvs_bench_grid_search_v2.py)
2. HNSW best configs not noted in the excel sheet with all_runs_summary