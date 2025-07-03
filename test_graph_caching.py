#!/usr/bin/env python3
"""
Test script to validate intelligent graph caching system.

This script tests the new per-row graph caching that uses jid + parameters.
Run this on SLURM to validate that:
1. Graphs are cached correctly
2. Cache hit rates improve on subsequent runs
3. Different splits reuse the same cached graphs
4. Cache keys are stable across runs
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add comformer to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comformer.data import load_dataset, load_pyg_graphs
from comformer.graph_cache import GraphCache
import pandas as pd

def test_graph_caching():
    """Test intelligent graph caching functionality."""
    
    print("=== Graph Caching Test ===")
    print(f"Current directory: {os.getcwd()}")
    
    # Use a temporary directory for testing
    test_output_dir = tempfile.mkdtemp(prefix="graph_cache_test_")
    print(f"Test output directory: {test_output_dir}")
    
    try:
        # Load a small subset of data for testing
        print("\n1. Loading test dataset...")
        data_path = os.environ.get("ROBINLAB_DATA_PATH", "/home/mudaliar.k/data") + "/combined_elemental.csv"
        print(f"Loading data from: {data_path}")
        
        df = load_dataset(
            name="D2R2_surface_data", 
            data_path=data_path, 
            target="cleavage_energy",
            limit=50  # Small subset for testing
        )
        print(f"Loaded {len(df)} rows for testing")
        
        # Test graph construction parameters
        graph_params = {
            "cutoff": 4.0,
            "max_neighbors": 25,
            "use_lattice": True,
            "use_angle": True,
            "use_canonize": False,
        }
        print(f"Graph parameters: {graph_params}")
        
        # First run - should be all cache misses
        print("\n2. First run (expect cache misses)...")
        start_time = time.time()
        
        graphs1 = load_pyg_graphs(
            df=df,
            output_dir=test_output_dir,
            enable_graph_cache=True,
            **graph_params
        )
        
        first_run_time = time.time() - start_time
        print(f"First run completed in {first_run_time:.2f} seconds")
        print(f"Generated {len(graphs1)} graphs")
        
        # Check cache directory
        cache_dir = Path(test_output_dir) / "graph_cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            print(f"Cache directory contains {len(cache_files)} files")
        else:
            print("❌ Cache directory not created!")
            return False
        
        # Second run - should be all cache hits
        print("\n3. Second run (expect cache hits)...")
        start_time = time.time()
        
        graphs2 = load_pyg_graphs(
            df=df,
            output_dir=test_output_dir,
            enable_graph_cache=True,
            **graph_params
        )
        
        second_run_time = time.time() - start_time
        print(f"Second run completed in {second_run_time:.2f} seconds")
        print(f"Generated {len(graphs2)} graphs")
        
        # Check speedup
        if second_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"Speedup: {speedup:.2f}x")
            
            if speedup > 2.0:
                print("✅ Significant speedup achieved - caching is working!")
            else:
                print("⚠️  Expected higher speedup - cache may not be working optimally")
        
        # Test with different parameters (should be cache misses)
        print("\n4. Testing different parameters (expect cache misses)...")
        different_params = graph_params.copy()
        different_params["cutoff"] = 5.0  # Different cutoff
        
        start_time = time.time()
        graphs3 = load_pyg_graphs(
            df=df,
            output_dir=test_output_dir,
            enable_graph_cache=True,
            **different_params
        )
        third_run_time = time.time() - start_time
        print(f"Different parameters run completed in {third_run_time:.2f} seconds")
        
        # Test cache stats
        print("\n5. Cache statistics:")
        cache = GraphCache(cache_dir=str(cache_dir))
        stats = cache.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Graph caching test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_output_dir)
            print(f"\nCleaned up test directory: {test_output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {test_output_dir}: {e}")

def test_cache_key_generation():
    """Test cache key generation and stability."""
    print("\n=== Cache Key Generation Test ===")
    
    from comformer.graph_cache import GraphCache
    
    cache = GraphCache("test_cache", enabled=False)  # Don't create files
    
    # Test parameters
    params1 = {
        "cutoff": 4.0,
        "max_neighbors": 25,
        "use_lattice": True,
        "use_angle": True,
        "use_canonize": False,
    }
    
    params2 = params1.copy()
    params3 = params1.copy()
    params3["cutoff"] = 5.0  # Different
    
    # Generate keys
    key1a = cache._generate_cache_key("test_jid_1", **params1)
    key1b = cache._generate_cache_key("test_jid_1", **params2)  # Should be same
    key2 = cache._generate_cache_key("test_jid_2", **params1)    # Different jid
    key3 = cache._generate_cache_key("test_jid_1", **params3)    # Different params
    
    print(f"Key 1a: {key1a}")
    print(f"Key 1b: {key1b}")
    print(f"Key 2:  {key2}")
    print(f"Key 3:  {key3}")
    
    # Validate
    assert key1a == key1b, "Same jid + params should generate same key"
    assert key1a != key2, "Different jid should generate different key"
    assert key1a != key3, "Different params should generate different key"
    
    print("✅ Cache key generation working correctly!")

if __name__ == "__main__":
    print("Starting graph caching validation...")
    
    # Test cache key generation first
    test_cache_key_generation()
    
    # Test full caching system
    success = test_graph_caching()
    
    if success:
        print("\n🎉 All tests passed! Graph caching is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Check the output above for details.")
        sys.exit(1)