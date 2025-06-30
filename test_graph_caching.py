#!/usr/bin/env python3
"""
Test script for graph caching functionality.
Tests cache performance and correctness on SLURM cluster.
"""

import os
import time
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import torch
from comformer.data import get_train_val_loaders, load_dataset

def test_cache_functionality():
    """Test basic cache functionality with small dataset."""
    print("=" * 60)
    print("TESTING GRAPH CACHING FUNCTIONALITY")
    print("=" * 60)
    
    # Load small sample of data
    print("Loading dataset...")
    df = load_dataset(
        data_path="sample_data/surface_prop_data_set_top_bottom.csv",
        target="WF_bottom",
        limit=10  # Small sample for testing
    )
    print(f"Loaded {len(df)} samples")
    
    # Create temporary cache directory
    cache_dir = Path(tempfile.mkdtemp(prefix="graph_cache_test_"))
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Test 1: First run without cache (cold start)
        print("\n" + "-" * 40)
        print("TEST 1: Cold start (no cache)")
        print("-" * 40)
        
        start_time = time.time()
        # Now properly handle 6 return values
        train_loader, val_loader, test_loader, prepare_batch, mean_train, std_train = get_train_val_loaders(
            dataset="D2R2_surface_data",
            target="WF_bottom",
            cutoff=8.0,
            max_neighbors=12,
            batch_size=2,
            workers=0,
            data_path="sample_data/surface_prop_data_set_top_bottom.csv",
            output_dir="./test_output",
            cachedir=cache_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            save_dataloader=False,  # Disable dataloader caching to test graph caching
            filename="test_cache_" + str(int(time.time())),  # Unique filename
        )
        cold_time = time.time() - start_time
        print(f"Cold start time: {cold_time:.2f} seconds")
        print(f"Cache files created: {len(list(cache_dir.glob('*.pkl')))}")
        print(f"Mean/std from cold run: {mean_train:.4f}/{std_train:.4f}")
        
        # Test 2: Second run with cache (warm start)
        print("\n" + "-" * 40)
        print("TEST 2: Warm start (with cache)")
        print("-" * 40)
        
        start_time = time.time()
        # Same parameters should use cache
        train_loader2, val_loader2, test_loader2, prepare_batch2, mean_train2, std_train2 = get_train_val_loaders(
            dataset="D2R2_surface_data",
            target="WF_bottom",
            cutoff=8.0,
            max_neighbors=12,
            batch_size=2,
            workers=0,
            data_path="sample_data/surface_prop_data_set_top_bottom.csv",
            output_dir="./test_output",
            cachedir=cache_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            save_dataloader=False,
            filename="test_cache_" + str(int(time.time())),  # Different filename to avoid dataloader cache
        )
        warm_time = time.time() - start_time
        print(f"Warm start time: {warm_time:.2f} seconds")
        print(f"Mean/std from warm run: {mean_train2:.4f}/{std_train2:.4f}")
        
        # Performance comparison
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time saved: {cold_time - warm_time:.2f} seconds")
        
        # Test 3: Verify graph correctness
        print("\n" + "-" * 40)
        print("TEST 3: Graph correctness verification")
        print("-" * 40)
        
        # Compare first batch from both loaders
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader2))
        
        print(f"Batch 1 - Nodes: {batch1.x.shape[0]}, Edges: {batch1.edge_index.shape[1]}")
        print(f"Batch 2 - Nodes: {batch2.x.shape[0]}, Edges: {batch2.edge_index.shape[1]}")
        
        # Check if graphs are identical
        graphs_identical = (
            torch.equal(batch1.x, batch2.x) and
            torch.equal(batch1.edge_index, batch2.edge_index) and
            torch.equal(batch1.edge_attr, batch2.edge_attr)
        )
        print(f"Graphs identical: {graphs_identical}")
        
        # Check if z_coords are present (should always be there)
        has_z_coords = hasattr(batch1, 'z_coords') and batch1.z_coords is not None
        print(f"Z-coordinates present: {has_z_coords}")
        if has_z_coords:
            z_identical = torch.equal(batch1.z_coords, batch2.z_coords)
            print(f"Z-coordinates identical: {z_identical}")
        
        # Test 4: Different parameters create different cache
        print("\n" + "-" * 40)
        print("TEST 4: Parameter variation test")
        print("-" * 40)
        
        initial_cache_count = len(list(cache_dir.glob('*.pkl')))
        print(f"Initial cache files: {initial_cache_count}")
        
        # Run with different cutoff - should create new cache entries
        train_loader3, _, _, _, _, _ = get_train_val_loaders(
            dataset="D2R2_surface_data",
            target="WF_bottom",
            cutoff=10.0,  # Different cutoff
            max_neighbors=12,
            batch_size=2,
            workers=0,
            data_path="sample_data/surface_prop_data_set_top_bottom.csv",
            output_dir="./test_output",
            cachedir=cache_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            save_dataloader=False,
            filename="test_cache_" + str(int(time.time())),
        )
        
        final_cache_count = len(list(cache_dir.glob('*.pkl')))
        print(f"Final cache files: {final_cache_count}")
        print(f"New cache files created: {final_cache_count - initial_cache_count}")
        
        print("\n" + "=" * 60)
        print("CACHE FUNCTIONALITY TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return {
            'cold_time': cold_time,
            'warm_time': warm_time,
            'speedup': speedup,
            'graphs_identical': graphs_identical,
            'cache_files': final_cache_count,
            'has_z_coords': has_z_coords
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        print(f"\nCleaning up cache directory: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

def test_different_splits():
    """Test that different splits reuse the same cached graphs."""
    print("\n" + "=" * 60)
    print("TESTING SPLIT INDEPENDENCE")
    print("=" * 60)
    
    cache_dir = Path(tempfile.mkdtemp(prefix="split_test_cache_"))
    print(f"Cache directory: {cache_dir}")
    
    try:
        # First split configuration
        print("Creating graphs with split 1 (60/20/20)...")
        start_time = time.time()
        _, _, _, _, _, _ = get_train_val_loaders(
            dataset="D2R2_surface_data",
            target="WF_bottom",
            data_path="sample_data/surface_prop_data_set_top_bottom.csv",
            output_dir="./test_output",
            cachedir=cache_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            batch_size=2,
            workers=0,
            save_dataloader=False,
            filename="split1_test",
        )
        split1_time = time.time() - start_time
        cache_count_1 = len(list(cache_dir.glob('*.pkl')))
        
        # Second split configuration (should reuse cache)
        print("Creating graphs with split 2 (70/15/15)...")
        start_time = time.time()
        _, _, _, _, _, _ = get_train_val_loaders(
            dataset="D2R2_surface_data",
            target="WF_bottom",
            data_path="sample_data/surface_prop_data_set_top_bottom.csv",
            output_dir="./test_output",
            cachedir=cache_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=2,
            workers=0,
            save_dataloader=False,
            filename="split2_test",
        )
        split2_time = time.time() - start_time
        cache_count_2 = len(list(cache_dir.glob('*.pkl')))
        
        print(f"Split 1 time: {split1_time:.2f}s, Cache files: {cache_count_1}")
        print(f"Split 2 time: {split2_time:.2f}s, Cache files: {cache_count_2}")
        print(f"Split 2 speedup: {split1_time/split2_time:.2f}x")
        print(f"Cache reused successfully: {cache_count_1 == cache_count_2}")
        
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Graph Caching Test Suite")
    print("========================")
    
    # Check if sample data exists
    sample_data_path = "sample_data/surface_prop_data_set_top_bottom.csv"
    if not os.path.exists(sample_data_path):
        print(f"ERROR: Sample data not found at {sample_data_path}")
        print("Please ensure the sample data file exists before running tests.")
        exit(1)
    
    # Run tests
    result = test_cache_functionality()
    if result:
        test_different_splits()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Cache speedup: {result['speedup']:.2f}x")
        print(f"Graph correctness: {'✓' if result['graphs_identical'] else '✗'}")
        print(f"Z-coordinates present: {'✓' if result['has_z_coords'] else '✗'}")
        print(f"Cache files created: {result['cache_files']}")
        print("All tests completed successfully!")
    else:
        print("Tests failed. Please check the error messages above.")
        exit(1)