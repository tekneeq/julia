#!/usr/bin/env python3
"""
Options Data Caching System

Provides local caching for Robinhood options data to reduce API calls
and improve performance during repeated analysis.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

class OptionsCache:
    """Manages local caching of options data."""
    
    def __init__(self, cache_dir: str = ".options_cache", default_expiry_hours: int = 4):
        """
        Initialize the options cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_expiry_hours: Default cache expiration time in hours
        """
        self.cache_dir = Path(cache_dir)
        self.default_expiry_hours = default_expiry_hours
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or create cache configuration
        self.config_file = self.cache_dir / "cache_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cache configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            config = {
                "created_at": datetime.now().isoformat(),
                "default_expiry_hours": self.default_expiry_hours,
                "total_requests_saved": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            self._save_config(config)
            return config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save cache configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_cache_key(self, ticker: str, expiration_date: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for options data."""
        # Include relevant parameters that affect the data
        key_data = {
            "ticker": ticker.upper(),
            "expiration_date": expiration_date,
            "params": params or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, ticker: str, expiration_date: str, params: Dict[str, Any] = None) -> Path:
        """Get file path for cached data."""
        ticker_dir = self.cache_dir / ticker.upper()
        ticker_dir.mkdir(exist_ok=True)
        
        cache_key = self._get_cache_key(ticker, expiration_date, params)
        filename = f"{expiration_date}_{cache_key}.json"
        return ticker_dir / filename
    
    def _is_cache_valid(self, cache_path: Path, expiry_hours: Optional[int] = None) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        expiry_hours = expiry_hours or self.default_expiry_hours
        
        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = mtime + timedelta(hours=expiry_hours)
        
        return datetime.now() < expiry_time
    
    def get(self, ticker: str, expiration_date: str, params: Dict[str, Any] = None, 
            expiry_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached options data.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Option expiration date (YYYY-MM-DD)
            params: Additional parameters that affect the data
            expiry_hours: Custom expiry time in hours
            
        Returns:
            Cached data if valid, None otherwise
        """
        cache_path = self._get_cache_path(ticker, expiration_date, params)
        
        if self._is_cache_valid(cache_path, expiry_hours):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Update stats
                self.config["cache_hits"] += 1
                self._save_config(self.config)
                
                print(f"📦 Using cached data for {ticker} {expiration_date}")
                return cached_data
            
            except (json.JSONDecodeError, FileNotFoundError):
                # Corrupted cache file
                cache_path.unlink(missing_ok=True)
        
        # Cache miss
        self.config["cache_misses"] += 1
        self._save_config(self.config)
        return None
    
    def set(self, ticker: str, expiration_date: str, data: Dict[str, Any], 
            params: Dict[str, Any] = None) -> None:
        """
        Cache options data.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Option expiration date (YYYY-MM-DD)
            data: Options data to cache
            params: Additional parameters that affect the data
        """
        cache_path = self._get_cache_path(ticker, expiration_date, params)
        
        # Add metadata to cached data
        cache_entry = {
            "ticker": ticker.upper(),
            "expiration_date": expiration_date,
            "cached_at": datetime.now().isoformat(),
            "params": params or {},
            "data": data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            
            # Update stats
            self.config["total_requests_saved"] += 1
            self._save_config(self.config)
            
            print(f"💾 Cached data for {ticker} {expiration_date}")
            
        except Exception as e:
            print(f"⚠️ Failed to cache data: {e}")
    
    def clear_ticker(self, ticker: str) -> int:
        """Clear all cached data for a specific ticker."""
        ticker_dir = self.cache_dir / ticker.upper()
        count = 0
        
        if ticker_dir.exists():
            for cache_file in ticker_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
            
            # Remove directory if empty
            try:
                ticker_dir.rmdir()
            except OSError:
                pass  # Directory not empty
        
        print(f"🗑️ Cleared {count} cache files for {ticker}")
        return count
    
    def clear_expired(self) -> int:
        """Clear all expired cache files."""
        count = 0
        
        for ticker_dir in self.cache_dir.iterdir():
            if ticker_dir.is_dir() and ticker_dir.name != "cache_config.json":
                for cache_file in ticker_dir.glob("*.json"):
                    if not self._is_cache_valid(cache_file):
                        cache_file.unlink()
                        count += 1
                
                # Remove directory if empty
                try:
                    ticker_dir.rmdir()
                except OSError:
                    pass  # Directory not empty
        
        print(f"🗑️ Cleared {count} expired cache files")
        return count
    
    def clear_all(self) -> int:
        """Clear all cached data."""
        count = 0
        
        for ticker_dir in self.cache_dir.iterdir():
            if ticker_dir.is_dir():
                for cache_file in ticker_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
                ticker_dir.rmdir()
        
        # Reset stats
        self.config["total_requests_saved"] = 0
        self.config["cache_hits"] = 0
        self.config["cache_misses"] = 0
        self._save_config(self.config)
        
        print(f"🗑️ Cleared all {count} cache files")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = []
        total_size = 0
        
        for ticker_dir in self.cache_dir.iterdir():
            if ticker_dir.is_dir():
                for cache_file in ticker_dir.glob("*.json"):
                    size = cache_file.stat().st_size
                    total_size += size
                    cache_files.append({
                        "ticker": ticker_dir.name,
                        "file": cache_file.name,
                        "size": size,
                        "modified": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat(),
                        "valid": self._is_cache_valid(cache_file)
                    })
        
        hit_rate = 0
        total_requests = self.config["cache_hits"] + self.config["cache_misses"]
        if total_requests > 0:
            hit_rate = (self.config["cache_hits"] / total_requests) * 100
        
        return {
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_hits": self.config["cache_hits"],
            "cache_misses": self.config["cache_misses"],
            "hit_rate_percent": round(hit_rate, 1),
            "requests_saved": self.config["total_requests_saved"],
            "files": cache_files
        }
    
    def list_cached_data(self) -> List[Dict[str, Any]]:
        """List all cached data with details."""
        cached_data = []
        
        for ticker_dir in self.cache_dir.iterdir():
            if ticker_dir.is_dir():
                for cache_file in ticker_dir.glob("*.json"):
                    try:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        
                        cached_data.append({
                            "ticker": data.get("ticker", ticker_dir.name),
                            "expiration_date": data.get("expiration_date", "unknown"),
                            "cached_at": data.get("cached_at", "unknown"),
                            "file_size": cache_file.stat().st_size,
                            "valid": self._is_cache_valid(cache_file),
                            "file_path": str(cache_file)
                        })
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue
        
        return sorted(cached_data, key=lambda x: x["cached_at"], reverse=True)

# Global cache instance
options_cache = OptionsCache()

def get_cache_instance() -> OptionsCache:
    """Get the global cache instance."""
    return options_cache