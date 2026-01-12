# Healthcare Feature Store

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![HIPAA Compliant](https://img.shields.io/badge/HIPAA-Compliant-green.svg)](#compliance)

**Production feature store for healthcare ML with point-in-time correct serving and HIPAA compliance.**

## 🎯 Business Impact

- **Point-in-time correctness** prevents data leakage in ML training
- **PHI-aware access control** with role-based feature access
- **Feature versioning** with full lineage
- **Online/offline consistency** for training-serving parity

## ✨ Key Features

- **Feature Registry**: Versioned feature definitions with lineage
- **Point-in-Time Serving**: Correct historical feature values
- **PHI Classification**: Direct, indirect, and non-PHI features
- **Access Control**: Role-based access with audit logging
- **Healthcare Categories**: Clinical, lab, medication, vital signs

## 🚀 Quick Start

```python
from src.registry.feature_registry import FeatureRegistry, FeatureCategory

registry = FeatureRegistry()

# Register a clinical feature
feature = registry.register_feature(
    name="hba1c_latest",
    value_type=FeatureValueType.FLOAT64,
    description="Most recent HbA1c lab value",
    category=FeatureCategory.LABORATORY,
    entity_type="patient",
    phi_level="indirect",
    access_roles=["clinician", "data_scientist"],
)
```

## 👤 Author

**Christopher Mangun** - [LinkedIn](https://linkedin.com/in/cmangun)
