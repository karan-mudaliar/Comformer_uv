"""
Intelligent per-row graph caching system.

Caches individual graphs based on jid + graph construction parameters.
This allows efficient reuse across different train/val/test splits.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger()

class GraphCache:
    """Per-row graph caching system using jid + parameters as key."""
    
    def __init__(self, cache_dir: str = "graph_cache", enabled: bool = True):
        """
        Initialize graph cache.
        
        Args:
            cache_dir: Directory to store cached graphs
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"Graph cache initialized at {self.cache_dir}")
    
    def _generate_cache_key(self, jid: str, **graph_params) -> str:
        """
        Generate stable cache key from jid and graph parameters.
        
        Args:
            jid: Unique identifier for the structure
            **graph_params: Graph construction parameters
            
        Returns:
            Stable cache key string
        """
        # Sort parameters for consistent ordering
        param_str = "_".join(f"{k}={v}" for k, v in sorted(graph_params.items()))
        cache_key = f"{jid}_{param_str}"
        
        # Hash long keys to avoid filesystem issues
        if len(cache_key) > 200:
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cached graph file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, jid: str, **graph_params) -> Optional[Any]:
        """
        Retrieve cached graph if available.
        
        Args:
            jid: Unique identifier for the structure
            **graph_params: Graph construction parameters
            
        Returns:
            Cached graph object or None if not found
        """
        if not self.enabled:
            return None
            
        cache_key = self._generate_cache_key(jid, **graph_params)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    graph = pickle.load(f)
                self.cache_hits += 1
                return graph
            except Exception as e:
                logger.warning(f"Failed to load cached graph {cache_key}: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
                
        self.cache_misses += 1
        return None
    
    def put(self, jid: str, graph: Any, **graph_params) -> None:
        """
        Store graph in cache.
        
        Args:
            jid: Unique identifier for the structure
            graph: Graph object to cache
            **graph_params: Graph construction parameters
        """
        if not self.enabled:
            return
            
        cache_key = self._generate_cache_key(jid, **graph_params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(graph, f)
        except Exception as e:
            logger.warning(f"Failed to cache graph {cache_key}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        cache_files = list(self.cache_dir.glob("*.pkl")) if self.cache_dir.exists() else []
        
        return {
            "enabled": self.enabled,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_graphs": len(cache_files),
            "cache_dir": str(self.cache_dir)
        }
    
    def clear(self) -> None:
        """Clear all cached graphs."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared graph cache at {self.cache_dir}")
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        logger.info(f"Graph cache stats: {stats['cache_hits']} hits, {stats['cache_misses']} misses, "
                   f"{stats['hit_rate']:.2%} hit rate, {stats['cached_graphs']} cached graphs")