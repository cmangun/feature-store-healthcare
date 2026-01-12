"""
Healthcare Feature Store - Feature Registry

Production feature registry for healthcare ML:
- Point-in-time correct feature serving
- Feature versioning and lineage
- Healthcare-specific validation
- HIPAA-compliant feature access
- Online/offline feature consistency
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, TypeVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class FeatureValueType(str, Enum):
    """Supported feature value types."""
    
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    ARRAY_INT = "array_int"
    ARRAY_FLOAT = "array_float"
    ARRAY_STRING = "array_string"
    EMBEDDING = "embedding"


class FeatureStatus(str, Enum):
    """Feature lifecycle status."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class FeatureCategory(str, Enum):
    """Healthcare feature categories."""
    
    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    LABORATORY = "laboratory"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"
    VITAL_SIGN = "vital_sign"
    IMAGING = "imaging"
    GENOMIC = "genomic"
    BEHAVIORAL = "behavioral"
    SOCIAL = "social"
    DERIVED = "derived"


@dataclass
class FeatureSchema:
    """Schema definition for a feature."""
    
    name: str
    value_type: FeatureValueType
    description: str
    category: FeatureCategory
    entity_type: str  # patient, encounter, provider, etc.
    is_nullable: bool = True
    default_value: Any = None
    validation_rules: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type.value,
            "description": self.description,
            "category": self.category.value,
            "entity_type": self.entity_type,
            "is_nullable": self.is_nullable,
            "default_value": self.default_value,
            "validation_rules": self.validation_rules,
        }


@dataclass
class FeatureSource:
    """Data source for a feature."""
    
    source_type: str  # batch, stream, derived
    source_location: str
    query: str | None = None
    transformation_logic: str | None = None
    refresh_frequency: str | None = None  # daily, hourly, realtime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_location": self.source_location,
            "query": self.query,
            "transformation_logic": self.transformation_logic,
            "refresh_frequency": self.refresh_frequency,
        }


@dataclass
class Feature:
    """A registered feature definition."""
    
    feature_id: str
    name: str
    version: str
    schema: FeatureSchema
    source: FeatureSource
    status: FeatureStatus
    owner: str
    created_at: datetime
    updated_at: datetime
    tags: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # feature_ids
    
    # Healthcare-specific
    phi_level: str = "none"  # none, indirect, direct
    access_roles: list[str] = field(default_factory=list)
    retention_days: int = 2555  # 7 years for HIPAA
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "version": self.version,
            "schema": self.schema.to_dict(),
            "source": self.source.to_dict(),
            "status": self.status.value,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "dependencies": self.dependencies,
            "phi_level": self.phi_level,
            "access_roles": self.access_roles,
            "retention_days": self.retention_days,
        }


@dataclass
class FeatureGroup:
    """A group of related features."""
    
    group_id: str
    name: str
    description: str
    entity_type: str
    features: list[str]  # feature_ids
    created_at: datetime
    owner: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "entity_type": self.entity_type,
            "feature_count": len(self.features),
            "created_at": self.created_at.isoformat(),
            "owner": self.owner,
        }


@dataclass
class FeatureValue:
    """A point-in-time feature value."""
    
    feature_id: str
    entity_id: str
    value: Any
    event_timestamp: datetime
    created_timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "entity_id": self.entity_id,
            "value": self.value,
            "event_timestamp": self.event_timestamp.isoformat(),
            "created_timestamp": self.created_timestamp.isoformat(),
        }


@dataclass
class FeatureVector:
    """A collection of feature values for an entity."""
    
    entity_id: str
    entity_type: str
    features: dict[str, Any]  # feature_name -> value
    timestamp: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
        }


class FeatureRegistryConfig(BaseModel):
    """Configuration for feature registry."""
    
    enable_versioning: bool = True
    enable_lineage: bool = True
    enable_validation: bool = True
    default_ttl_hours: int = Field(default=24, ge=1)
    max_feature_age_days: int = Field(default=30, ge=1)
    
    # Healthcare-specific
    require_phi_classification: bool = True
    require_access_roles: bool = True
    audit_all_access: bool = True


class FeatureRegistry:
    """
    Healthcare Feature Registry.
    
    Features:
    - Feature definition and versioning
    - Point-in-time correct serving
    - Dependency tracking
    - Healthcare compliance (PHI classification)
    - Access control integration
    """
    
    def __init__(self, config: FeatureRegistryConfig | None = None):
        self.config = config or FeatureRegistryConfig()
        self._features: dict[str, Feature] = {}
        self._groups: dict[str, FeatureGroup] = {}
        self._values: dict[str, list[FeatureValue]] = defaultdict(list)  # feature_id -> values
        self._access_log: list[dict[str, Any]] = []
    
    def register_feature(
        self,
        name: str,
        value_type: FeatureValueType,
        description: str,
        category: FeatureCategory,
        entity_type: str,
        source: FeatureSource,
        owner: str,
        version: str = "1.0.0",
        phi_level: str = "none",
        access_roles: list[str] | None = None,
        **kwargs: Any,
    ) -> Feature:
        """
        Register a new feature.
        
        Args:
            name: Feature name
            value_type: Data type
            description: Feature description
            category: Healthcare category
            entity_type: Entity type (patient, encounter, etc.)
            source: Data source information
            owner: Feature owner
            version: Feature version
            phi_level: PHI classification
            access_roles: Roles allowed to access
        
        Returns:
            Registered Feature
        """
        # Validate PHI classification if required
        if self.config.require_phi_classification and phi_level not in ["none", "indirect", "direct"]:
            raise ValueError(f"Invalid PHI level: {phi_level}")
        
        # Require access roles for PHI features
        if self.config.require_access_roles and phi_level != "none" and not access_roles:
            raise ValueError("Access roles required for PHI features")
        
        feature_id = self._generate_feature_id(name, version, entity_type)
        now = datetime.utcnow()
        
        schema = FeatureSchema(
            name=name,
            value_type=value_type,
            description=description,
            category=category,
            entity_type=entity_type,
            **{k: v for k, v in kwargs.items() if k in FeatureSchema.__dataclass_fields__},
        )
        
        feature = Feature(
            feature_id=feature_id,
            name=name,
            version=version,
            schema=schema,
            source=source,
            status=FeatureStatus.DRAFT,
            owner=owner,
            created_at=now,
            updated_at=now,
            phi_level=phi_level,
            access_roles=access_roles or [],
            tags=kwargs.get("tags", {}),
        )
        
        self._features[feature_id] = feature
        
        logger.info(
            "feature_registered",
            feature_id=feature_id,
            name=name,
            category=category.value,
            phi_level=phi_level,
        )
        
        return feature
    
    def activate_feature(self, feature_id: str) -> Feature:
        """Activate a feature for use."""
        feature = self._features.get(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        feature.status = FeatureStatus.ACTIVE
        feature.updated_at = datetime.utcnow()
        
        logger.info("feature_activated", feature_id=feature_id)
        return feature
    
    def deprecate_feature(self, feature_id: str, reason: str) -> Feature:
        """Deprecate a feature."""
        feature = self._features.get(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        feature.status = FeatureStatus.DEPRECATED
        feature.updated_at = datetime.utcnow()
        feature.tags["deprecation_reason"] = reason
        
        logger.warning("feature_deprecated", feature_id=feature_id, reason=reason)
        return feature
    
    def create_feature_group(
        self,
        name: str,
        description: str,
        entity_type: str,
        feature_ids: list[str],
        owner: str,
    ) -> FeatureGroup:
        """Create a feature group."""
        # Validate all features exist and have same entity type
        for fid in feature_ids:
            feature = self._features.get(fid)
            if not feature:
                raise ValueError(f"Feature not found: {fid}")
            if feature.schema.entity_type != entity_type:
                raise ValueError(
                    f"Feature {fid} has entity type {feature.schema.entity_type}, "
                    f"expected {entity_type}"
                )
        
        group_id = self._generate_group_id(name, entity_type)
        
        group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            entity_type=entity_type,
            features=feature_ids,
            created_at=datetime.utcnow(),
            owner=owner,
        )
        
        self._groups[group_id] = group
        
        logger.info(
            "feature_group_created",
            group_id=group_id,
            name=name,
            feature_count=len(feature_ids),
        )
        
        return group
    
    def ingest_feature_value(
        self,
        feature_id: str,
        entity_id: str,
        value: Any,
        event_timestamp: datetime,
    ) -> FeatureValue:
        """
        Ingest a feature value.
        
        Args:
            feature_id: Feature to update
            entity_id: Entity identifier
            value: Feature value
            event_timestamp: When the value was observed
        
        Returns:
            Ingested FeatureValue
        """
        feature = self._features.get(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        if feature.status != FeatureStatus.ACTIVE:
            raise ValueError(f"Feature {feature_id} is not active")
        
        # Validate value type
        if self.config.enable_validation:
            self._validate_value(value, feature.schema.value_type)
        
        feature_value = FeatureValue(
            feature_id=feature_id,
            entity_id=entity_id,
            value=value,
            event_timestamp=event_timestamp,
            created_timestamp=datetime.utcnow(),
        )
        
        # Store value (append for point-in-time correctness)
        self._values[feature_id].append(feature_value)
        
        return feature_value
    
    def get_feature_value(
        self,
        feature_id: str,
        entity_id: str,
        as_of: datetime | None = None,
        user_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> FeatureValue | None:
        """
        Get a feature value with point-in-time correctness.
        
        Args:
            feature_id: Feature to retrieve
            entity_id: Entity identifier
            as_of: Point-in-time (defaults to now)
            user_id: User requesting access
            user_roles: User's roles for access control
        
        Returns:
            Most recent FeatureValue as of timestamp, or None
        """
        feature = self._features.get(feature_id)
        if not feature:
            raise ValueError(f"Feature not found: {feature_id}")
        
        # Access control check
        if self.config.require_access_roles and feature.phi_level != "none":
            if not user_roles or not any(r in feature.access_roles for r in user_roles):
                self._log_access_denied(feature_id, entity_id, user_id)
                raise PermissionError(
                    f"Access denied to feature {feature_id}. "
                    f"Required roles: {feature.access_roles}"
                )
        
        as_of = as_of or datetime.utcnow()
        
        # Find most recent value before as_of
        values = self._values.get(feature_id, [])
        matching = [
            v for v in values
            if v.entity_id == entity_id and v.event_timestamp <= as_of
        ]
        
        if not matching:
            return None
        
        # Return most recent
        result = max(matching, key=lambda v: v.event_timestamp)
        
        # Audit logging
        if self.config.audit_all_access:
            self._log_access(feature_id, entity_id, user_id, as_of)
        
        return result
    
    def get_feature_vector(
        self,
        entity_id: str,
        entity_type: str,
        feature_ids: list[str],
        as_of: datetime | None = None,
        user_id: str | None = None,
        user_roles: list[str] | None = None,
    ) -> FeatureVector:
        """
        Get multiple features for an entity.
        
        Returns a FeatureVector with all requested feature values.
        """
        as_of = as_of or datetime.utcnow()
        features: dict[str, Any] = {}
        
        for feature_id in feature_ids:
            try:
                value = self.get_feature_value(
                    feature_id=feature_id,
                    entity_id=entity_id,
                    as_of=as_of,
                    user_id=user_id,
                    user_roles=user_roles,
                )
                feature = self._features[feature_id]
                features[feature.name] = value.value if value else None
            except PermissionError:
                # Skip features user can't access
                features[self._features[feature_id].name] = None
        
        return FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            timestamp=as_of,
        )
    
    def list_features(
        self,
        category: FeatureCategory | None = None,
        entity_type: str | None = None,
        status: FeatureStatus | None = None,
        phi_level: str | None = None,
    ) -> list[Feature]:
        """List features with filters."""
        features = list(self._features.values())
        
        if category:
            features = [f for f in features if f.schema.category == category]
        if entity_type:
            features = [f for f in features if f.schema.entity_type == entity_type]
        if status:
            features = [f for f in features if f.status == status]
        if phi_level:
            features = [f for f in features if f.phi_level == phi_level]
        
        return features
    
    def _validate_value(self, value: Any, value_type: FeatureValueType) -> None:
        """Validate value matches expected type."""
        if value is None:
            return  # Nullable by default
        
        type_checks = {
            FeatureValueType.INT32: lambda v: isinstance(v, int),
            FeatureValueType.INT64: lambda v: isinstance(v, int),
            FeatureValueType.FLOAT32: lambda v: isinstance(v, (int, float)),
            FeatureValueType.FLOAT64: lambda v: isinstance(v, (int, float)),
            FeatureValueType.STRING: lambda v: isinstance(v, str),
            FeatureValueType.BOOL: lambda v: isinstance(v, bool),
            FeatureValueType.ARRAY_INT: lambda v: isinstance(v, list),
            FeatureValueType.ARRAY_FLOAT: lambda v: isinstance(v, list),
            FeatureValueType.EMBEDDING: lambda v: isinstance(v, list),
        }
        
        check = type_checks.get(value_type)
        if check and not check(value):
            raise ValueError(f"Invalid value type for {value_type.value}")
    
    def _log_access(
        self,
        feature_id: str,
        entity_id: str,
        user_id: str | None,
        as_of: datetime,
    ) -> None:
        """Log feature access for audit."""
        self._access_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "access",
            "feature_id": feature_id,
            "entity_id": entity_id,
            "user_id": user_id,
            "as_of": as_of.isoformat(),
        })
    
    def _log_access_denied(
        self,
        feature_id: str,
        entity_id: str,
        user_id: str | None,
    ) -> None:
        """Log denied access for audit."""
        self._access_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "access_denied",
            "feature_id": feature_id,
            "entity_id": entity_id,
            "user_id": user_id,
        })
        logger.warning(
            "feature_access_denied",
            feature_id=feature_id,
            entity_id=entity_id,
            user_id=user_id,
        )
    
    def _generate_feature_id(
        self,
        name: str,
        version: str,
        entity_type: str,
    ) -> str:
        """Generate unique feature ID."""
        content = f"{name}:{version}:{entity_type}"
        return f"feat_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def _generate_group_id(self, name: str, entity_type: str) -> str:
        """Generate unique group ID."""
        content = f"{name}:{entity_type}"
        return f"grp_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
