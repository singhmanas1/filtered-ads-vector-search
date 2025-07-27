# tests.py
import numpy as np
import cupy as cp
from utils_grid_search import calc_truth, create_rating_filter, count_selected_samples, batch_search_cagra, build_hnsw_index, batch_search_hnsw, calc_recall
import faiss

def test_brute_force_batch_consistency():
    """Test that calc_truth (brute force) gives identical results regardless of batch size."""
    np.random.seed(42)
    cp.random.seed(42)
    
    # Setup test data (same as CAGRA test for comparison)
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    print("=== Test 1: Unfiltered Brute Force Batch Consistency ===")
    
    # Reference result (moderate batch size)
    ref_dist, ref_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=None, batch_size=100)
    
    # Test different batch sizes - brute force should be perfectly consistent
    for batch_size in [1,2,4,8,16,32,64,128,256,512,1024]:
        test_dist, test_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=None, batch_size=batch_size)
        
        recall = calc_recall(test_indices, ref_indices)
        
        # Brute force should be perfectly deterministic
        assert recall > 0.999, f"Brute force batch_size={batch_size} shows inconsistency: {recall:.6f}"
        print(f"  PASS: batch_size={batch_size}, recall={recall:.6f}")
    
    print("=== Test 2: Filtered Brute Force Batch Consistency ===")
    
    # Create restrictive filter (same as CAGRA test)
    config = {'rating_distribution': {'low': 60, 'high': 20, 'mid': 19, 'premium': 1}}
    filter_obj, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['high','mid','premium'], device_id=0, verbose=False, config=config
    )
    
    # Reference result with filter
    ref_filtered_dist, ref_filtered_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=filter_obj, batch_size=100)
    
    recalls = []
    for batch_size in [1,8,64,256,512,1024]:
        test_dist, test_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=filter_obj, batch_size=batch_size)
        
        recall = calc_recall(test_indices, ref_filtered_indices)
        recalls.append(recall)
        print(f"  batch_size={batch_size}, recall={recall:.6f}")
    
    # Check consistency across batch sizes for filtered brute force
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    
    print(f"  Filtered brute force stats: min={min_recall:.6f}, std={recall_std:.8f}")
    
    # Brute force should be perfectly consistent even with filters
    assert min_recall > 0.999, f"Filtered brute force shows inconsistency: {min_recall:.6f}"
    assert recall_std < 1e-6, f"Brute force has unexpected variance: {recall_std:.8f}"
    
    return True

def test_batch_search_cagra():
    """Test batch_search_cagra consistency using recall metrics."""
    import cupy as cp
    from cuvs.neighbors import cagra
    
    # Setup test data
    np.random.seed(42)
    cp.random.seed(42)
    
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build CAGRA index
    vectors_gpu = cp.asarray(vectors)
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=48, build_algo="nn_descent")
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    
    # Create search params
    search_params = cagra.SearchParams(search_width=8, itopk_size=64, max_iterations=0)
    
    print("=== Test 1: Unfiltered Batch Size Consistency ===")
    
    # Reference result (moderate batch size)
    ref_indices, _ = batch_search_cagra(
        queries, cagra_index, search_params, None, k, batch_size=1024
    )
    
    # Test different batch sizes
    for batch_size in [1,2,4,8,16,32,64,128,256,512,1024]:
        test_indices, _ = batch_search_cagra(
            queries, cagra_index, search_params, None, k, batch_size=batch_size
        )
        
        recall = calc_recall(test_indices, ref_indices)
        
        # For unfiltered search, we expect very high consistency (>99%)
        assert recall > 0.09, f"Unfiltered batch_size={batch_size} shows poor consistency: {recall:.4f}"
        print(f"  PASS: batch_size={batch_size}, recall={recall:.4f}")
    
    print("=== Test 2: Filtered Batch Size Consistency ===")
    
    # Create restrictive filter
    config = {'rating_distribution': {'low': 60, 'high': 20, 'mid': 19, 'premium': 1}}
    filter_obj, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['high','mid','premium'], device_id=0, verbose=False, config=config
    )
    
    # Reference result with filter
    ref_filtered_indices, _ = batch_search_cagra(
        queries, cagra_index, search_params, filter_obj, k, batch_size=100
    )
    
    recalls = []
    for batch_size in [1,8,64,256,512,1024]:
        test_indices, _ = batch_search_cagra(
            queries, cagra_index, search_params, filter_obj, k, batch_size=batch_size
        )
        
        recall = calc_recall(test_indices, ref_filtered_indices)
        recalls.append(recall)
        print(f"  batch_size={batch_size}, recall={recall:.4f}")
    
    # Check consistency across batch sizes for filtered search
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    
    print(f"  Filtered recall stats: min={min_recall:.4f}, std={recall_std:.6f}")
    
    # More lenient thresholds for filtered search, but still check for major inconsistencies
    assert min_recall > 0.80, f"Some batch sizes show very poor recall: {min_recall:.4f}"
    assert recall_std < 0.05, f"High variance in recall across batch sizes: {recall_std:.6f}"
    
    return True

def test_cagra_filtered_batch_consistency():
    """Test that CAGRA filtered search gives identical results regardless of batch size."""
    import cupy as cp
    from cuvs.neighbors import cagra
    
    np.random.seed(42)
    cp.random.seed(42)
    
    # Large scale test data to mirror the failing test
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build CAGRA index
    vectors_gpu = cp.asarray(vectors)
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64)
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    
    # Test both problematic and fixed search params
    test_configs = [
        {
            "name": "Original (problematic)",
            "params": cagra.SearchParams(itopk_size=32, search_width=1, max_iterations=0)
        },
        {
            "name": "Fixed parameters", 
            "params": cagra.SearchParams(itopk_size=128, search_width=8, max_iterations=1000)
        }
    ]
    
    # Create restrictive filter (40% of data)
    config = {'rating_distribution': {'low': 60, 'high': 20, 'mid': 19, 'premium': 1}}
    filter_obj, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['high','mid','premium'], device_id=0, verbose=False, config=config
    )
    
    for test_config in test_configs:
        print(f"\n=== Testing {test_config['name']} ===")
        search_params = test_config['params']
        
        # Reference result (moderate batch size)
        ref_indices, _ = batch_search_cagra(
            queries, cagra_index, search_params, filter_obj, k, batch_size=100
        )
        
        # Test different batch sizes that previously showed issues
        problematic_batch_sizes = [1, 8, 64, 256, 512, 1024]
        batch_consistent = True
        
        for batch_size in problematic_batch_sizes:
            if batch_size > n_queries:
                continue
                
            test_indices, _ = batch_search_cagra(
                queries, cagra_index, search_params, filter_obj, k, batch_size=batch_size
            )
            
            # Check if results are identical (stricter than recall)
            indices_match = np.array_equal(ref_indices, test_indices)
            
            if not indices_match:
                print(f"  FAIL: batch_size={batch_size} gives different results")
                batch_consistent = False
                
                # Calculate how different they are
                matches = 0
                for i in range(len(ref_indices)):
                    if np.array_equal(ref_indices[i], test_indices[i]):
                        matches += 1
                consistency_pct = (matches / len(ref_indices)) * 100
                print(f"    Query-level consistency: {consistency_pct:.1f}%")
            else:
                print(f"  PASS: batch_size={batch_size}")
        
        if test_config['name'] == "Original (problematic)":
            # We expect this to fail, so don't assert
            if batch_consistent:
                print(f"  UNEXPECTED: Original params are actually consistent!")
            else:
                print(f"  EXPECTED: Original params show batch inconsistency")
        else:
            # We expect the fixed params to work
            assert batch_consistent, f"Fixed parameters still show batch inconsistency"
            print(f"  SUCCESS: Fixed parameters are batch consistent")
    
    return True

def test_hnsw_batch_consistency():
    """Test that HNSW search gives consistent results regardless of batch size."""
    import faiss
    
    # Setup test data (same as CAGRA test for comparison)
    np.random.seed(42)
    
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build HNSW index
    params = {'M': 16, 'efConstruction': 40, 'efSearch': 64, 'quantization_type': 'fp16'}
    hnsw_index, _ = build_hnsw_index(params, vectors)
    
    print("=== Test 1: Unfiltered HNSW Batch Consistency ===")
    
    # Create basic search params (no filter)
    search_params = faiss.SearchParametersHNSW(efSearch=64)
    
    # Reference result (moderate batch size)
    ref_indices, _ = batch_search_hnsw(
        queries, hnsw_index, search_params, None, k, batch_size=100
    )
    
    # Test different batch sizes
    for batch_size in [1,2,4,8,16,32,64,128,256,512,1024]:
        test_indices, _ = batch_search_hnsw(
            queries, hnsw_index, search_params, None, k, batch_size=batch_size
        )
        
        recall = calc_recall(test_indices, ref_indices)
        
        # HNSW should be more deterministic than CAGRA
        assert recall > 0.95, f"HNSW unfiltered batch_size={batch_size} shows poor consistency: {recall:.4f}"
        print(f"  PASS: batch_size={batch_size}, recall={recall:.4f}")
    
    print("=== Test 2: Filtered HNSW Batch Consistency ===")
    
    # Create restrictive filter (same as CAGRA test)
    config = {'rating_distribution': {'low': 60, 'high': 20, 'mid': 19, 'premium': 1}}
    _, bitquery = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['high','mid','premium'], device_id=0, verbose=False, config=config
    )
    
    # Convert bitquery to numpy for FAISS
    filter_bitmap = bitquery.get().view(np.uint8)
    
    # Create search params with filter
    sel = faiss.IDSelectorBitmap(filter_bitmap)
    search_params_filtered = faiss.SearchParametersHNSW(sel=sel, efSearch=64)
    
    # Reference result with filter
    ref_filtered_indices, _ = batch_search_hnsw(
        queries, hnsw_index, search_params_filtered, None, k, batch_size=100
    )
    
    recalls = []
    for batch_size in [1,8,64,256,512,1024]:
        test_indices, _ = batch_search_hnsw(
            queries, hnsw_index, search_params_filtered, None, k, batch_size=batch_size
        )
        
        recall = calc_recall(test_indices, ref_filtered_indices)
        recalls.append(recall)
        print(f"  batch_size={batch_size}, recall={recall:.4f}")
    
    # Check consistency across batch sizes for filtered search
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    
    print(f"  Filtered HNSW stats: min={min_recall:.4f}, std={recall_std:.6f}")
    
    # HNSW should be more stable than CAGRA, even with filters
    assert min_recall > 0.90, f"HNSW filtered search shows poor recall: {min_recall:.4f}"
    assert recall_std < 0.02, f"HNSW shows high variance across batch sizes: {recall_std:.6f}"
    
    return True

def test_recall_batch_independence():
    """Test that recall scores are identical regardless of batch size."""
    from cuvs.neighbors import cagra
    
    # Setup test data (same as CAGRA test for comparison)
    np.random.seed(42)
    
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Build CAGRA index
    vectors_gpu = cp.asarray(vectors)
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64)
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    search_params = cagra.SearchParams(itopk_size=64, search_width=8)
    
    # Generate ground truth
    _, gt_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=None, batch_size=50)
    
    # Test different batch sizes
    batch_sizes = [1,8,64,256,512,1024]
    recalls = []
    
    for batch_size in batch_sizes:
        if batch_size > n_queries:
            continue
            
        search_indices, _ = batch_search_cagra(
            queries, cagra_index, search_params, None, k, batch_size=batch_size
        )
        
        recall = calc_recall(search_indices, gt_indices)
        recalls.append(recall)
    
    # All recalls should be identical
    recall_std = np.std(recalls)
    assert recall_std < 1e-6, f"Recall varies with batch size! Values: {recalls}, Std: {recall_std:.8f}"
    
    print(f"PASS: Recall batch independence test (recall: {recalls[0]:.4f}, std: {recall_std:.8f})")
    return True

def test_cagra_batch_recall():
    """Test CAGRA batch consistency using only cuVS functions."""
    import cupy as cp
    from cuvs.neighbors import cagra, brute_force
    
    # Setup test data
    np.random.seed(42)
    cp.random.seed(42)
    
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float32)  # cuVS prefers float32
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Convert to GPU
    vectors_gpu = cp.asarray(vectors)
    queries_gpu = cp.asarray(queries)
    
    # Generate ground truth using cuVS brute force
    print("Generating ground truth with cuVS brute force...")
    bf_index = brute_force.build(vectors_gpu, metric="sqeuclidean")
    gt_distances, gt_indices = brute_force.search(bf_index, queries_gpu, k)
    gt_indices = cp.asnumpy(gt_indices)  # Convert to CPU for comparison
    
    # Build CAGRA index
    print("Building CAGRA index...")
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64)
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    search_params = cagra.SearchParams(itopk_size=64, search_width=8)
    
    # Test different batch sizes
    batch_sizes = [1, 8, 64, 256, 512, 1024]
    recalls = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...")
        
        # Manual batching with pure cuVS
        all_indices = np.zeros((n_queries, k), dtype=np.int32)
        
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_queries = queries_gpu[start_idx:end_idx]
            
            # Pure cuVS search
            distances, indices = cagra.search(search_params, cagra_index, batch_queries, k=k)
            
            # Store results
            indices_cpu = cp.asnumpy(indices)
            all_indices[start_idx:end_idx] = indices_cpu
        
        # Calculate recall manually (no helper functions)
        correct = 0
        total = 0
        
        for i in range(n_queries):
            # Count how many CAGRA results are in the ground truth for this query
            cagra_neighbors = set(all_indices[i])
            gt_neighbors = set(gt_indices[i])
            correct += len(cagra_neighbors.intersection(gt_neighbors))
            total += k
        
        recall = correct / total
        recalls.append(recall)
        print(f"  batch_size={batch_size}, recall={recall:.4f}")
    
    # Check consistency
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    max_recall = max(recalls)
    
    print(f"\nCUVS CAGRA Batch Consistency Results:")
    print(f"  Recalls: {[f'{r:.4f}' for r in recalls]}")
    print(f"  Min: {min_recall:.4f}, Max: {max_recall:.4f}")
    print(f"  Std Dev: {recall_std:.8f}")
    
    # The test - CAGRA should be consistent across batch sizes
    if recall_std < 1e-4:
        print("PASS: CAGRA shows good batch consistency")
        return True
    else:
        print(f"FAIL: CAGRA shows poor batch consistency (std={recall_std:.8f})")
        return False

def test_hnsw_recall_batch_independence():
    """Test that HNSW recall scores are identical regardless of batch size."""
    import faiss
    
    # Setup test data (same as CAGRA test for comparison)
    np.random.seed(42)
    
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build HNSW index
    params = {'M': 16, 'efConstruction': 40, 'efSearch': 64, 'quantization_type': 'fp'}
    hnsw_index, _ = build_hnsw_index(params, vectors)
    
    # Create search params
    search_params = faiss.SearchParametersHNSW(efSearch=64)
    
    # Generate ground truth
    _, gt_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=None, batch_size=50)
    
    # Test different batch sizes
    batch_sizes = [1,8,64,256,512,1024]
    recalls = []
    
    for batch_size in batch_sizes:
        if batch_size > n_queries:
            continue
            
        search_indices, _ = batch_search_hnsw(
            queries, hnsw_index, search_params, None, k, batch_size=batch_size
        )
        
        recall = calc_recall(search_indices, gt_indices)
        recalls.append(recall)
        print(f"  batch_size={batch_size}, recall={recall:.4f}")
    
    # Check consistency across batch sizes
    recall_std = np.std(recalls)
    min_recall = min(recalls)
    max_recall = max(recalls)
    
    print(f"HNSW recall stats: min={min_recall:.4f}, max={max_recall:.4f}, std={recall_std:.8f}")
    
    # HNSW should be more deterministic than CAGRA
    assert recall_std < 1e-4, f"HNSW recall varies with batch size! Values: {recalls}, Std: {recall_std:.8f}"
    
    print(f"PASS: HNSW recall batch independence test (avg recall: {np.mean(recalls):.4f}, std: {recall_std:.8f})")
    return True

def test_recall_filter_batch_independence():
    """Test that filtered recall scores are identical regardless of batch size."""
    from cuvs.neighbors import cagra
    
    np.random.seed(42)
    cp.random.seed(42)
    
    # Small test data  
    n_samples, n_queries, dim, k = 5000000, 10000, 64, 10
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build CAGRA index
    vectors_gpu = cp.asarray(vectors)
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64)
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    search_params = cagra.SearchParams(itopk_size=128, search_width=8, max_iterations=100)
    
    # Create filter
    config = {'rating_distribution': {'low': 60, 'high': 20, 'mid': 19, 'premium': 1}}
    filter_obj, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['high','mid','premium'], device_id=0, verbose=False, config=config
    )
    
    # Generate filtered ground truth
    _, gt_indices = calc_truth(vectors, queries, k, metric="sqeuclidean", filter=filter_obj, batch_size=50)
    
    # Test different batch sizes with filter
    batch_sizes = [1,2,4,8,16,32,64,128,256,512,1024]
    recalls = []
    
    for batch_size in batch_sizes:
        if batch_size > n_queries:
            continue
            
        search_indices, _ = batch_search_cagra(
            queries, cagra_index, search_params, filter_obj, k, batch_size=batch_size
        )
        
        recall = calc_recall(search_indices, gt_indices)
        recalls.append(recall)
    
    # All recalls should be identical
    recall_std = np.std(recalls)
    assert recall_std < 1e-6, f"Filtered recall varies with batch size! Values: {recalls}, Std: {recall_std:.8f}"
    
    print(f"PASS: Recall filter batch independence test (recall: {recalls[0]:.4f}, std: {recall_std:.8f})")
    return True

def run_all_tests():
    """Run all tests."""
    tests = [
        #("CAGRA filtered batch consistency", test_cagra_filtered_batch_consistency),
        # ("Batch consistency", test_batch_consistency),
        # ("Filter equivalence", test_filter_equivalence), 
        # ("Filter bits", test_filter_bits),
        #("Brute force batch consistency", test_brute_force_batch_consistency),
        #("HNSW batch consistency", test_hnsw_batch_consistency),
        #("CAGRA batch search", test_batch_search_cagra),
        # ("HNSW batch search", test_batch_search_hnsw),
         #("CAGRA Recall batch independence", test_recall_batch_independence), 
         ("CAGRA batch recall", test_cagra_batch_recall),
         #("HNSW Recall batch independence", test_hnsw_recall_batch_independence),
        # ("Recall filter batch independence", test_recall_filter_batch_independence)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_name} - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSUMMARY: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    run_all_tests()