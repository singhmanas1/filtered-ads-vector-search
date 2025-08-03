# Amazon Reviews Vector Search Benchmark

A comprehensive benchmarking tool for NVIDIA CUVS CAGRA with Amazon review embeddings and filtering capabilities.

## Setup

### 1. Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Amazon Dataset
```bash
cd download_dataset
python download_amazon_dataset.py
```

### 3. Generate Embeddings

**Start embedding server:**
```bash
bash embedder_run.sh
```

**Generate embeddings (in another terminal):**
```bash
python exec_embed.py \
    --input_dir ./amazon_reviews_2023_categories/raw/review_categories \
    --output_dir ../amazon_review_embeddings \
    --categories "Books" "Baby Products" "Kindle Store" \
    --max_embeddings 100000 \
    --min_tokens 20 \
    --fp16
```

**Test embeddings:**
```bash
python embedding_test.py
```

## Data Structure
```
amazon_review_embeddings/
├── Books/
│ ├── embeddings_0.npy # 10K embeddings per file
│ ├── texts_0.npy # Corresponding texts
│ └── text_sources.npy # Category mapping
├── Baby Products/
└── Kindle Store/
```

## Run Benchmark

### Basic Usage
```bash
python cuvs_bench_grid_search_v3.py \
    --quantization-folder half_precision \
    --quantization-folder-path ./amazon_review_embeddings
```

### Background Execution
```bash
nohup python cuvs_bench_grid_search_v2.py \
    --quantization-folder half_precision \
    --quantization-folder-path ./amazon_review_embeddings \
    > benchmark.log 2>&1 &
```

## Key Files

| File | Purpose |
|------|---------|
| `download_amazon_dataset.py` | Download Amazon review data |
| `embedder_run.sh` | Start NVIDIA NIM embedding server |
| `exec_embed.py` | Generate embeddings from reviews |
| `embedding_test.py` | Validate embedding quality |
| `cuvs_bench_grid_search_v2.py` | Main benchmark script |

## Configuration

- **Categories**: Specify which Amazon categories to process
- **Max embeddings**: Limit total embeddings generated
- **Min tokens**: Filter reviews by minimum word count
- **FP16**: Use half-precision to save space

## Output

Results saved in `comprehensive_comparison_results/`:
- `all_runs_summary.csv` - Performance summary
- `results_*.csv` - Detailed metrics
- `comprehensive_grid_search_comparison.log` - Execution log

## Monitoring
```bash
# Watch progress
tail -f comprehensive_comparison_results/comprehensive_grid_search_comparison.log

# Check embeddings
ls amazon_review_embeddings/*/
```
```
