"""Cache management for pickle files and experiment results."""

import os
import pickle
from typing import List, Optional, Any
from pathlib import Path


class CacheManager:
    """Manages pickle cache files for expensive computations."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache manager with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def exists(self, cache_key: str) -> bool:
        """Check if cached result exists."""
        return self.get_cache_path(cache_key).exists()
    
    def load(self, cache_key: str) -> Optional[Any]:
        """Load cached result if it exists."""
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear specific cache or all caches."""
        if cache_key:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


# Convenience functions for common cache operations
def load_or_compute(cache_key: str, compute_func, cache_dir: str = "cache", force_recompute: bool = False):
    """Load from cache or compute and cache the result."""
    cache_manager = CacheManager(cache_dir)
    
    if not force_recompute and cache_manager.exists(cache_key):
        return cache_manager.load(cache_key)
    
    # Compute and cache
    result = compute_func()
    cache_manager.save(cache_key, result)
    return result