#!/usr/bin/env python3
"""
Test graph caching by running actual training on 1000 rows for 1 epoch, twice.

First run: No cache (clean start)
Second run: Should use cached graphs (much faster graph loading)
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add comformer to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comformer.train_props import train_prop_model

def test_caching_with_training():
    """Test caching by running actual training twice."""
    
    print("=== Graph Caching Training Test ===")
    print("Testing with 1000 rows, 1 epoch, iComformer cleavage energy")
    print("First run: Cold start (no cache)")
    print("Second run: Warm start (should use cached graphs)")
    print(f"Data path: {os.environ.get('ROBINLAB_DATA_PATH', '/home/mudaliar.k/data')}")
    
    # Create test output directories
    output_dir_1 = "output/cache_test_run1"
    output_dir_2 = "output/cache_test_run2" 
    
    # Clean up any existing test directories
    for dir_path in [output_dir_1, output_dir_2]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # Shared training parameters
    training_params = {
        "dataset": "D2R2_surface_data",
        "prop": "cleavage_energy", 
        "name": "iComformer",
        "pyg_input": True,
        "n_epochs": 1,  # Just 1 epoch for testing
        "max_neighbors": 25,
        "cutoff": 4.0,
        "batch_size": 32,
        "use_lattice": True,
        "use_angle": True,
        "break_z_symmetry": False,
        "save_dataloader": False,  # We disabled this
        "output_features": 1,
        "limit": 1000,  # Limit to 1000 rows for testing
        "data_path": os.environ.get("ROBINLAB_DATA_PATH", "/home/mudaliar.k/data") + "/combined_elemental.csv"
    }
    
    print(f"\nTraining parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    try:
        # === FIRST RUN (No cache) ===
        print(f"\n🚀 FIRST RUN - No cache available")
        print(f"Output directory: {output_dir_1}")
        
        start_time_1 = time.time()
        
        train_prop_model(
            **training_params,
            output_dir=output_dir_1
        )
        
        total_time_1 = time.time() - start_time_1
        print(f"✅ First run completed in {total_time_1:.2f} seconds")
        
        # Check if cache was created
        cache_dir = Path(output_dir_1) / "graph_cache"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            print(f"📦 Cache created with {len(cache_files)} graph files")
        else:
            print("❌ No cache directory found!")
            return False
        
        # === SECOND RUN (With cache) ===
        print(f"\n🚀 SECOND RUN - Should use cached graphs")
        print(f"Output directory: {output_dir_2}")
        
        # Copy cache from first run to second run
        cache_dir_2 = Path(output_dir_2) / "graph_cache"
        shutil.copytree(cache_dir, cache_dir_2)
        print(f"📋 Copied cache with {len(list(cache_dir_2.glob('*.pkl')))} files")
        
        start_time_2 = time.time()
        
        train_prop_model(
            **training_params,
            output_dir=output_dir_2
        )
        
        total_time_2 = time.time() - start_time_2
        print(f"✅ Second run completed in {total_time_2:.2f} seconds")
        
        # === ANALYSIS ===
        print(f"\n📊 PERFORMANCE ANALYSIS")
        print(f"First run (no cache):  {total_time_1:.2f} seconds")
        print(f"Second run (cached):   {total_time_2:.2f} seconds")
        
        if total_time_2 > 0:
            speedup = total_time_1 / total_time_2
            time_saved = total_time_1 - total_time_2
            percent_saved = (time_saved / total_time_1) * 100
            
            print(f"Time saved: {time_saved:.2f} seconds ({percent_saved:.1f}%)")
            print(f"Speedup: {speedup:.2f}x")
            
            if speedup > 1.2:
                print("✅ Caching provided measurable speedup!")
            else:
                print("⚠️  Speedup less than expected - cache may not be working optimally")
            
            if speedup > 2.0:
                print("🎉 Excellent speedup! Graph caching is working very well!")
        
        # Check final cache state
        final_cache_files = list(cache_dir_2.glob("*.pkl"))
        print(f"\n📦 Final cache contains {len(final_cache_files)} graph files")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test directories (optional - comment out to inspect results)
        print(f"\n🧹 Cleaning up test directories...")
        for dir_path in [output_dir_1, output_dir_2]:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"  Removed {dir_path}")
            except Exception as e:
                print(f"  Warning: Could not remove {dir_path}: {e}")

if __name__ == "__main__":
    print("Starting graph caching training test...")
    print("This will run iComformer training on 1000 rows for 1 epoch, twice.")
    print("The second run should be faster due to cached graphs.")
    
    success = test_caching_with_training()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("Graph caching is working correctly if second run was faster.")
    else:
        print("\n💥 Test failed. Check the output above for details.")
    
    sys.exit(0 if success else 1)