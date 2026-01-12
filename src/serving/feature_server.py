"""
Feature Serving Layer for Healthcare Feature Store.

Production-grade feature serving for ML models:
- Online serving with low-latency retrieval
- Offline batch serving for training
- Point-in-time correct feature retrieval
- Feature freshness monitoring
- Caching with configurable TTL
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from feature_store_healthcare.src.registry.feature_registry import (
    FeatureRegistry,
    Feature,
    FeatureGroup,
)

logger = structlog.get_logger()


class ServingMode(Enum):
    """Feature serving modes."""
    
    ONLINE = "online"     # Low-latency, single entity
    OFFLINE = "offline"   # Batch, historical
    STREAMING = "streaming"  # Real-time updates


class FeatureFreshness(Enum):
    """Feature freshness status."""
    
    FRESH = "fresh"       # Within SLA
    STALE = "stale"       # Exceeds SLA but usable
    EXPIRED = "expired"   # Too old, may cause issues


@dataclass
class FeatureValue:
    """Single feature value with metadata."""
    
    feature_name: str
    value: Any
    timestamp: datetime
    freshness: FeatureFreshness
    version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "freshness": self.freshness.value,
            "version": self.version,
        }


@dataclass
class FeatureVector:
    """Collection of features for an entity."""
    
    entity_id: str
    entity_type: str
    features: dict[str, FeatureValue]
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    cache_hit: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "retrieved_at": self.retrieved_at.isoformat(),
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
        }
    
    def to_flat_dict(self) -> dict[str, Any]:
        """Get flat dictionary of feature values for ML inference."""
        return {
            name: fv.value
            for name, fv in self.features.items()
        }


@dataclass
class ServingConfig:
    """Feature serving configuration."""
    
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 10000
    online_timeout_ms: int = 100
    offline_batch_size: int = 1000
    freshness_sla_seconds: int = 3600  # 1 hour
    stale_threshold_seconds: int = 86400  # 24 hours


@dataclass
class ServingMetrics:
    """Feature serving metrics."""
    
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    stale_features_served: int = 0
    errors: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.total_requests, 1),
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "stale_features_served": self.stale_features_served,
            "errors": self.errors,
        }


class LRUCache:
    """LRU cache for feature vectors."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[FeatureVector, datetime]] = OrderedDict()
    
    def get(self, key: str, ttl_seconds: int) -> FeatureVector | None:
        """Get value if exists and not expired."""
        if key not in self.cache:
            return None
        
        vector, cached_at = self.cache[key]
        
        # Check TTL
        if (datetime.utcnow() - cached_at).total_seconds() > ttl_seconds:
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return vector
    
    def put(self, key: str, vector: FeatureVector) -> None:
        """Add value to cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                self.cache.popitem(last=False)
        
        self.cache[key] = (vector, datetime.utcnow())
    
    def invalidate(self, key: str) -> None:
        """Remove key from cache."""
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()


class FeatureServer:
    """
    Production feature serving layer.
    
    Features:
    - Online serving with caching
    - Offline batch retrieval
    - Point-in-time correct lookups
    - Freshness monitoring
    - Latency tracking
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        config: ServingConfig | None = None,
    ):
        self.registry = registry
        self.config = config or ServingConfig()
        self.cache = LRUCache(self.config.cache_max_size)
        self.metrics = ServingMetrics()
        self._latencies: list[float] = []
        
        # In-memory feature store (replace with Redis/DynamoDB in production)
        self._online_store: dict[str, dict[str, Any]] = {}
        self._offline_store: list[dict[str, Any]] = []
    
    def get_online_features(
        self,
        entity_id: str,
        entity_type: str,
        feature_names: list[str],
    ) -> FeatureVector:
        """
        Get features for a single entity with low latency.
        
        Args:
            entity_id: Entity identifier (e.g., patient_id)
            entity_type: Entity type (e.g., "patient")
            feature_names: List of feature names to retrieve
            
        Returns:
            Feature vector with values and metadata
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        cache_key = self._cache_key(entity_id, entity_type, feature_names)
        
        # Try cache first
        if self.config.cache_enabled:
            cached = self.cache.get(cache_key, self.config.cache_ttl_seconds)
            if cached:
                self.metrics.cache_hits += 1
                latency = (time.time() - start_time) * 1000
                cached.latency_ms = latency
                cached.cache_hit = True
                self._record_latency(latency)
                return cached
        
        self.metrics.cache_misses += 1
        
        # Fetch from online store
        try:
            features = self._fetch_online_features(
                entity_id,
                entity_type,
                feature_names,
            )
            
            vector = FeatureVector(
                entity_id=entity_id,
                entity_type=entity_type,
                features=features,
                cache_hit=False,
            )
            
            # Track stale features
            stale_count = sum(
                1 for f in features.values()
                if f.freshness in [FeatureFreshness.STALE, FeatureFreshness.EXPIRED]
            )
            self.metrics.stale_features_served += stale_count
            
            # Cache result
            if self.config.cache_enabled:
                self.cache.put(cache_key, vector)
            
            latency = (time.time() - start_time) * 1000
            vector.latency_ms = latency
            self._record_latency(latency)
            
            logger.debug(
                "online_features_served",
                entity_id=entity_id,
                feature_count=len(features),
                latency_ms=latency,
                cache_hit=False,
            )
            
            return vector
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(
                "online_feature_fetch_error",
                entity_id=entity_id,
                error=str(e),
            )
            raise
    
    def get_offline_features(
        self,
        entity_ids: list[str],
        entity_type: str,
        feature_names: list[str],
        event_timestamp: datetime | None = None,
    ) -> list[FeatureVector]:
        """
        Get features for multiple entities (batch mode).
        
        Args:
            entity_ids: List of entity identifiers
            entity_type: Entity type
            feature_names: Features to retrieve
            event_timestamp: Point-in-time for historical features
            
        Returns:
            List of feature vectors
        """
        start_time = time.time()
        results: list[FeatureVector] = []
        
        # Process in batches
        for i in range(0, len(entity_ids), self.config.offline_batch_size):
            batch = entity_ids[i:i + self.config.offline_batch_size]
            
            for entity_id in batch:
                try:
                    features = self._fetch_offline_features(
                        entity_id,
                        entity_type,
                        feature_names,
                        event_timestamp,
                    )
                    
                    results.append(FeatureVector(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        features=features,
                    ))
                    
                except Exception as e:
                    logger.warning(
                        "offline_feature_fetch_error",
                        entity_id=entity_id,
                        error=str(e),
                    )
                    # Return empty vector for failed entities
                    results.append(FeatureVector(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        features={},
                    ))
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(
            "offline_features_served",
            entity_count=len(entity_ids),
            feature_count=len(feature_names),
            total_time_ms=total_time,
        )
        
        return results
    
    def get_point_in_time_features(
        self,
        entity_df: list[dict[str, Any]],
        feature_names: list[str],
        entity_column: str = "entity_id",
        timestamp_column: str = "event_timestamp",
    ) -> list[dict[str, Any]]:
        """
        Get point-in-time correct features for training data.
        
        Ensures no data leakage by only using features available
        at the time of each event.
        
        Args:
            entity_df: List of entities with timestamps
            feature_names: Features to join
            entity_column: Column containing entity IDs
            timestamp_column: Column containing event timestamps
            
        Returns:
            Entity data enriched with features
        """
        results: list[dict[str, Any]] = []
        
        for row in entity_df:
            entity_id = row.get(entity_column)
            event_time = row.get(timestamp_column)
            
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time)
            
            # Get features as of event time
            features = self._fetch_offline_features(
                entity_id=str(entity_id),
                entity_type="entity",
                feature_names=feature_names,
                as_of_time=event_time,
            )
            
            # Merge with entity data
            result = dict(row)
            for name, fv in features.items():
                result[name] = fv.value
                result[f"{name}__timestamp"] = fv.timestamp.isoformat()
            
            results.append(result)
        
        logger.info(
            "point_in_time_features_generated",
            row_count=len(results),
            feature_count=len(feature_names),
        )
        
        return results
    
    def write_features(
        self,
        entity_id: str,
        entity_type: str,
        features: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> None:
        """
        Write feature values to the store.
        
        Args:
            entity_id: Entity identifier
            entity_type: Entity type
            features: Feature name-value pairs
            timestamp: Feature timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.utcnow()
        
        store_key = f"{entity_type}:{entity_id}"
        
        if store_key not in self._online_store:
            self._online_store[store_key] = {}
        
        for name, value in features.items():
            self._online_store[store_key][name] = {
                "value": value,
                "timestamp": timestamp.isoformat(),
                "version": 1,
            }
        
        # Also write to offline store for historical access
        self._offline_store.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            **features,
            "_timestamp": timestamp.isoformat(),
        })
        
        # Invalidate cache
        self.cache.invalidate(store_key)
        
        logger.debug(
            "features_written",
            entity_id=entity_id,
            feature_count=len(features),
        )
    
    def get_metrics(self) -> ServingMetrics:
        """Get serving metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset serving metrics."""
        self.metrics = ServingMetrics()
        self._latencies.clear()
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _cache_key(
        self,
        entity_id: str,
        entity_type: str,
        feature_names: list[str],
    ) -> str:
        """Generate cache key."""
        features_str = ",".join(sorted(feature_names))
        content = f"{entity_type}:{entity_id}:{features_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _record_latency(self, latency_ms: float) -> None:
        """Record latency for metrics."""
        self._latencies.append(latency_ms)
        
        # Keep last 1000 for P99 calculation
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        
        # Update metrics
        self.metrics.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        sorted_latencies = sorted(self._latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        self.metrics.p99_latency_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
    
    def _fetch_online_features(
        self,
        entity_id: str,
        entity_type: str,
        feature_names: list[str],
    ) -> dict[str, FeatureValue]:
        """Fetch features from online store."""
        store_key = f"{entity_type}:{entity_id}"
        stored = self._online_store.get(store_key, {})
        
        features: dict[str, FeatureValue] = {}
        
        for name in feature_names:
            if name in stored:
                data = stored[name]
                timestamp = datetime.fromisoformat(data["timestamp"])
                freshness = self._calculate_freshness(timestamp)
                
                features[name] = FeatureValue(
                    feature_name=name,
                    value=data["value"],
                    timestamp=timestamp,
                    freshness=freshness,
                    version=data.get("version", 1),
                )
            else:
                # Return None for missing features
                features[name] = FeatureValue(
                    feature_name=name,
                    value=None,
                    timestamp=datetime.utcnow(),
                    freshness=FeatureFreshness.EXPIRED,
                )
        
        return features
    
    def _fetch_offline_features(
        self,
        entity_id: str,
        entity_type: str,
        feature_names: list[str],
        as_of_time: datetime | None = None,
    ) -> dict[str, FeatureValue]:
        """Fetch features from offline store with point-in-time support."""
        features: dict[str, FeatureValue] = {}
        
        # Filter to entity
        entity_rows = [
            row for row in self._offline_store
            if row.get("entity_id") == entity_id
            and row.get("entity_type") == entity_type
        ]
        
        # Apply time filter if specified
        if as_of_time:
            entity_rows = [
                row for row in entity_rows
                if datetime.fromisoformat(row["_timestamp"]) <= as_of_time
            ]
        
        # Get latest value for each feature
        for name in feature_names:
            latest_row = None
            latest_time = None
            
            for row in entity_rows:
                if name in row:
                    row_time = datetime.fromisoformat(row["_timestamp"])
                    if latest_time is None or row_time > latest_time:
                        latest_time = row_time
                        latest_row = row
            
            if latest_row and name in latest_row:
                freshness = self._calculate_freshness(latest_time)
                features[name] = FeatureValue(
                    feature_name=name,
                    value=latest_row[name],
                    timestamp=latest_time,
                    freshness=freshness,
                )
            else:
                features[name] = FeatureValue(
                    feature_name=name,
                    value=None,
                    timestamp=datetime.utcnow(),
                    freshness=FeatureFreshness.EXPIRED,
                )
        
        return features
    
    def _calculate_freshness(self, timestamp: datetime) -> FeatureFreshness:
        """Calculate feature freshness based on timestamp."""
        age_seconds = (datetime.utcnow() - timestamp).total_seconds()
        
        if age_seconds <= self.config.freshness_sla_seconds:
            return FeatureFreshness.FRESH
        elif age_seconds <= self.config.stale_threshold_seconds:
            return FeatureFreshness.STALE
        else:
            return FeatureFreshness.EXPIRED
