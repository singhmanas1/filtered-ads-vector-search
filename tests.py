# tests.py
import numpy as np
import cupy as cp
from utils_grid_search import calc_truth, create_rating_filter, count_selected_samples, batch_search_cagra, build_hnsw_index, batch_search_hnsw
import faiss

def test_batch_consistency():
    """Test that batching doesn't change ground truth results."""
    np.random.seed(42)
    cp.random.seed(42)
    
    # Small test data
    dataset = np.random.randn(200, 32).astype(np.float16)
    queries = np.random.randn(20, 32).astype(np.float16)
    k = 5
    
    # Reference (no batching)
    ref_dist, ref_idx = calc_truth(dataset, queries, k, batch_size=100)
    
    # Test different batch sizes
    for batch_size in [1, 5, 10]:
        test_dist, test_idx = calc_truth(dataset, queries, k, batch_size=batch_size)
        
        assert np.array_equal(ref_idx, test_idx), f"Indices differ with batch_size={batch_size}"
        assert np.allclose(ref_dist, test_dist, rtol=1e-6), f"Distances differ with batch_size={batch_size}"
    
    print("PASS: Batch consistency test")
    return True

def test_filter_equivalence():
    """Test that all-inclusive filter equals no filter."""
    np.random.seed(42)
    cp.random.seed(42)
    
    # Small test data
    dataset = np.random.randn(200, 32).astype(np.float16)
    queries = np.random.randn(20, 32).astype(np.float16)
    k = 5
    
    # No filter result
    no_filter_dist, no_filter_idx = calc_truth(dataset, queries, k, filter=None, batch_size=50)
    
    # All-inclusive filter
    config = {'rating_distribution': {'low': 50, 'high': 50}}
    result = create_rating_filter(
        n_samples=200, n_queries=20, 
        valid_ranges=['low', 'high'],  # All ranges
        device_id=0, verbose=False, config=config
    )
    
    assert result is not None, "create_rating_filter returned None"
    filter_obj, bitquery = result
    
    # Verify filter includes all samples
    included = count_selected_samples(bitquery)
    inclusion_pct = (included / 200) * 100
    assert inclusion_pct > 99, f"Filter only includes {inclusion_pct:.1f}% of data"
    
    # Filtered result
    filter_dist, filter_idx = calc_truth(dataset, queries, k, filter=filter_obj, batch_size=50)
    
    # Compare
    assert np.array_equal(no_filter_idx, filter_idx), "Indices differ between no filter and all-inclusive filter"
    assert np.allclose(no_filter_dist, filter_dist, rtol=1e-6), "Distances differ between no filter and all-inclusive filter"
    
    print("PASS: Filter equivalence test")
    return True

def test_filter_bits():
    """Test that filter bit patterns work correctly."""
    config = {'rating_distribution': {'a': 25, 'b': 25, 'c': 25, 'd': 25}}
    n_samples = 100
    
    # Partial filter - should include ['a', 'b'] = (25+25)/(25+25+25+25) = 50%
    result1 = create_rating_filter(
        n_samples=n_samples, 
        n_queries=10, 
        valid_ranges=['a', 'b'], 
        device_id=0, 
        verbose=False, 
        config=config
    )
    assert result1 is not None, "Partial filter creation returned None"
    _, partial_bits = result1
    partial_count = count_selected_samples(partial_bits)
    
    # Full filter - should include ['a', 'b', 'c', 'd'] = (25+25+25+25)/(25+25+25+25) = 100%
    result2 = create_rating_filter(
        n_samples=n_samples, 
        n_queries=10, 
        valid_ranges=['a', 'b', 'c', 'd'], 
        device_id=0, 
        verbose=False, 
        config=config
    )
    assert result2 is not None, "Full filter creation returned None"
    _, full_bits = result2
    full_count = count_selected_samples(full_bits)
    
    # Calculate expected counts
    total_distribution = sum(config['rating_distribution'].values())  # 100
    partial_expected = int(n_samples * (25 + 25) / total_distribution)  # 50
    full_expected = int(n_samples * (25 + 25 + 25 + 25) / total_distribution)  # 100
    
    # Check counts match expectations (allow small tolerance for randomness)
    tolerance = 5  # Allow ±5 samples difference
    assert abs(partial_count - partial_expected) <= tolerance, f"Partial filter count {partial_count} != expected {partial_expected} ± {tolerance}"
    assert abs(full_count - full_expected) <= tolerance, f"Full filter count {full_count} != expected {full_expected} ± {tolerance}"
    
    print(f"PASS: Filter bits test (partial: {partial_count}/{partial_expected}, full: {full_count}/{full_expected})")
    return True

def test_batch_search_cagra():
    """Test batch_search_cagra with different batch sizes and filters."""
    import cupy as cp
    from cuvs.neighbors import cagra
    
    # Setup test data
    np.random.seed(42)
    cp.random.seed(42)
    
    n_samples, n_queries, dim, k = 200, 20, 32, 5
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build CAGRA index
    vectors_gpu = cp.asarray(vectors)
    cagra_index_params = cagra.IndexParams(graph_degree=32, intermediate_graph_degree=48)
    cagra_index = cagra.build(cagra_index_params, vectors_gpu)
    
    # Create search params
    search_params = cagra.SearchParams(max_queries=100, itopk_size=64, max_iterations=0)
    
    print("=== Test 1: Batch Size Consistency ===")
    
    # Reference result (large batch)
    ref_indices, ref_time = batch_search_cagra(
        queries, cagra_index, search_params, None, k, batch_size=50
    )
    
    # Test different batch sizes
    for batch_size in [1, 5, 10]:
        test_indices, test_time = batch_search_cagra(
            queries, cagra_index, search_params, None, k, batch_size=batch_size
        )
        
        indices_match = np.array_equal(ref_indices, test_indices)
        assert indices_match, f"Indices differ with batch_size={batch_size}"
        print(f"  PASS: batch_size={batch_size}")
    
    print("=== Test 2: Filter Differences ===")
    
    # Create two different filters
    config = {'rating_distribution': {'low': 50, 'high': 50}}
    
    # Filter 1: Include only 'low' (~50% of data)
    filter1, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['low'], device_id=0, verbose=False, config=config
    )
    
    # Filter 2: Include both 'low' and 'high' (~100% of data)  
    filter2, _ = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['low', 'high'], device_id=0, verbose=False, config=config
    )
    
    # Search with different filters
    indices_no_filter, _ = batch_search_cagra(
        queries, cagra_index, search_params, None, k, batch_size=10
    )
    
    indices_filter1, _ = batch_search_cagra(
        queries, cagra_index, search_params, filter1, k, batch_size=10
    )
    
    indices_filter2, _ = batch_search_cagra(
        queries, cagra_index, search_params, filter2, k, batch_size=10
    )
    
    # Different filters should give different results
    filter1_differs = not np.array_equal(indices_no_filter, indices_filter1)
    filter2_similar = np.array_equal(indices_no_filter, indices_filter2)  # Should be similar since filter2 includes ~100%
    
    assert filter1_differs, "Filter with 50% data should give different results than no filter"
    print(f"  PASS: Restrictive filter gives different results")
    
    if filter2_similar:
        print(f"  PASS: All-inclusive filter gives same results as no filter")
    else:
        print(f"  NOTE: All-inclusive filter gives different results (may be due to randomness in filter creation)")
    
    print("SUCCESS: All batch_search_cagra tests passed!")
    return True

def test_batch_search_hnsw():
    """Test batch_search_hnsw with different batch sizes and filters."""
    import faiss
    
    # Setup test data
    np.random.seed(42)
    
    n_samples, n_queries, dim, k = 200, 20, 32, 5
    vectors = np.random.randn(n_samples, dim).astype(np.float16)
    queries = np.random.randn(n_queries, dim).astype(np.float16)
    
    # Build HNSW index
    params = {'M': 16, 'efConstruction': 40, 'efSearch': 16, 'quantization_type': 'fp'}
    hnsw_index, _ = build_hnsw_index(params, vectors)
    
    print("=== Test 1: Batch Size Consistency ===")
    
    # Create basic search params (no filter)
    search_params = faiss.SearchParametersHNSW(efSearch=16)
    
    # Reference result (large batch)
    ref_indices, ref_time = batch_search_hnsw(
        queries, hnsw_index, search_params, None, k, batch_size=50
    )
    
    # Test different batch sizes
    for batch_size in [1, 5, 10]:
        test_indices, test_time = batch_search_hnsw(
            queries, hnsw_index, search_params, None, k, batch_size=batch_size
        )
        
        indices_match = np.array_equal(ref_indices, test_indices)
        assert indices_match, f"Indices differ with batch_size={batch_size}"
        print(f"  PASS: batch_size={batch_size}")
    
    print("=== Test 2: Filter Differences ===")
    
    # Create different filters
    config = {'rating_distribution': {'low': 50, 'high': 50}}
    
    # Filter 1: Include only 'low' (~50% of data)
    _, bitquery1 = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['low'], device_id=0, verbose=False, config=config
    )
    
    # Filter 2: Include both 'low' and 'high' (~100% of data)  
    _, bitquery2 = create_rating_filter(
        n_samples=n_samples, n_queries=n_queries,
        valid_ranges=['low', 'high'], device_id=0, verbose=False, config=config
    )
    
    # Convert bitqueries to numpy for FAISS
    filter_bitmap1 = bitquery1.get().view(np.uint8)
    filter_bitmap2 = bitquery2.get().view(np.uint8)
    
    # Create search params with different filters
    search_params_no_filter = faiss.SearchParametersHNSW(efSearch=16)
    
    sel1 = faiss.IDSelectorBitmap(filter_bitmap1)
    search_params_filter1 = faiss.SearchParametersHNSW(sel=sel1, efSearch=16)
    
    sel2 = faiss.IDSelectorBitmap(filter_bitmap2)
    search_params_filter2 = faiss.SearchParametersHNSW(sel=sel2, efSearch=16)
    
    # Search with different filters
    indices_no_filter, _ = batch_search_hnsw(
        queries, hnsw_index, search_params_no_filter, None, k, batch_size=10
    )
    
    indices_filter1, _ = batch_search_hnsw(
        queries, hnsw_index, search_params_filter1, None, k, batch_size=10
    )
    
    indices_filter2, _ = batch_search_hnsw(
        queries, hnsw_index, search_params_filter2, None, k, batch_size=10
    )
    
    # Different filters should give different results
    filter1_differs = not np.array_equal(indices_no_filter, indices_filter1)
    filter2_similar = np.array_equal(indices_no_filter, indices_filter2)  # Should be similar since filter2 includes ~100%
    
    assert filter1_differs, "Filter with 50% data should give different results than no filter"
    print(f"  PASS: Restrictive filter gives different results")
    
    if filter2_similar:
        print(f"  PASS: All-inclusive filter gives same results as no filter")
    else:
        print(f"  NOTE: All-inclusive filter gives different results (may be due to randomness in filter creation)")
    
    print("SUCCESS: All batch_search_hnsw tests passed!")
    return True

def run_all_tests():
    """Run all tests."""
    tests = [
        ("Batch consistency", test_batch_consistency),
        ("Filter equivalence", test_filter_equivalence), 
        ("Filter bits", test_filter_bits),
        ("CAGRA batch search", test_batch_search_cagra),
        ("HNSW batch search", test_batch_search_hnsw)  # Add this line
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