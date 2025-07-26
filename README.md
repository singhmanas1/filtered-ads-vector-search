# CAGRA Grid Search Benchmark

A comprehensive benchmarking tool for NVIDIA CUVS CAGRA (Cuda Approximate Graph-based Retrieval Algorithm) with support for multiple quantization types and parameter optimization.

## Overview

This tool performs systematic grid search evaluation across:
- Multiple data sizes (1M, 5M, 10M+ vectors)
- Various quantization types (half precision, full precision, scalar, binary)
- Different filtering configurations (rating-based filtering)
- Comprehensive parameter sweeps for optimal performance tuning

## Installation

```bash
# Create virtual environment
python -m venv cagra_benchmark_env

# Activate virtual environment
source cagra_benchmark_env/bin/activate
```

### Install Dependencies
```bash
# Install dependencies
pip install -r requirements.txt
```

## Data Setup

### Option 1: AWS S3 Bucket (Alternative)

If you don't have access to the Google Drive folder, you can access the data from the [AWS S3 bucket](https://us-east-1.console.aws.amazon.com/s3/buckets/amazon-ads-vector-search?region=us-east-1&tab=objects&bucketType=general).

1. Download the raw data from the S3 bucket
2. Use the provided `index_data.ipynb` notebook to convert the data to embeddings format:

This will process the raw data and create the required `embeddings/` folder structure.

### Option 2: Download from Google Drive (If you don't have access to the AWS S3 Bucket)

1. Download the sample data from [Google Drive](https://drive.google.com/drive/folders/1PlvcajPYOrpjAGd4i2mUOO2H0MJc6zIr)
   - `sample_1M.tar.gz` (309.4 MB) - 1M vectors sample
   - `sample_6M.tar.gz` (1.81 GB) - 6M vectors sample

2. Extract the data:

```bash
# For 1M sample
tar -xzf sample_1M.tar.gz

# For 6M sample  
tar -xzf sample_6M.tar.gz
```

This will create the `embeddings/half_precision/` folder structure with the required `.npy` files.



## Quick Start

### Basic Usage
```bash
python cuvs_bench_grid_search_v2.py \
    --quantization-folder half_precision \
    --quantization-folder-path /path/to/embeddings \
    --config-path .
```

### All Quantization Types
```bash
# Half precision
python cuvs_bench_grid_search_v2.py --quantization-folder half_precision --quantization-folder-path /path/to/embeddings

# Full precision  
python cuvs_bench_grid_search_v2.py --quantization-folder full_precision --quantization-folder-path /path/to/embeddings

# Scalar quantization
python cuvs_bench_grid_search_v2.py --quantization-folder scalar --quantization-folder-path /path/to/embeddings

# Binary quantization
python cuvs_bench_grid_search_v2.py --quantization-folder binary --quantization-folder-path /path/to/embeddings
```

### Background Execution
```bash
nohup python cuvs_bench_grid_search_v2.py \
    --quantization-folder half_precision \
    --quantization-folder-path /path/to/embeddings \
    > benchmark.log 2>&1 &
```

## Required Files

### Python Files
- `cuvs_bench_grid_search_v2.py` (main script)
- `utils_grid_search.py` (utility functions)

### Configuration Files
- `filter_config.yaml` (filter configurations)
- `cagra_config.yaml` (CAGRA parameters)

### Data Structure

embeddings/
├── half_precision/
│ ├── 000000_fp16.npy
│ └── 000001_fp16.npy
├── full_precision/
│  ├── 000000_full.npy
│  └── 000001_full.npy
└── [other quantization types]/

## Monitoring

```bash
# Monitor execution log
tail -f comprehensive_grid_search_comparison.log
```

## Output Files

All results are saved in the `comprehensive_comparison_results/` directory.

### Key Files

| File | Description |
|------|-------------|
| `all_runs_summary.csv` | High-level summary of all benchmark runs |
| `comprehensive_grid_search_comparison.log` | Detailed execution log |
| `results_[size]vecs_[filter].csv` | Detailed results per configuration |

### File Contents

#### Summary File (`all_runs_summary.csv`)
- **data_size**: Vector count (1M, 5M, etc.)
- **filter_name**: Rating filter applied
- **best_recall**: Highest accuracy achieved
- **best_qps**: Peak queries per second
- **run_time_minutes**: Total execution time

#### Detailed Results (`results_*.csv`)
- **Parameters**: `graph_degree`, `search_width`, `itopk_size`, etc.
- **Performance**: `queries_per_second`, `latency_ms`, `build_time_seconds`
- **Quality**: `recall` (search accuracy)
- **Dataset**: `total_vectors`, `total_queries`

### Monitoring Progress

```bash
# Watch execution log
tail -f comprehensive_comparison_results/comprehensive_grid_search_comparison.log

# Check summary
cat comprehensive_comparison_results/all_runs_summary.csv

# List all results
ls comprehensive_comparison_results/results_*.csv
```

## Visualize Results-

Run `create_plots.ipynb`