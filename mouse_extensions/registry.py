"""
Mouse Extensions - Registry

Central registry for datasets and models.
Provides clean separation between original GS-LRM and mouse extensions.
"""

from typing import Dict, Type, Any


# Dataset Registry (lazy loaded)
_DATASET_REGISTRY: Dict[str, Type] = {}
_INITIALIZED = False


def _ensure_initialized():
    """Lazily initialize the registry to avoid circular imports."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    
    # Register mouse datasets
    from mouse_extensions.data.mouse_dataset import MouseViewDataset
    _DATASET_REGISTRY["mouse"] = MouseViewDataset
    _DATASET_REGISTRY["MouseViewDataset"] = MouseViewDataset


def register_dataset(name: str):
    """Decorator to register a dataset class."""
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_class(name: str) -> Type:
    """Get dataset class by name."""
    _ensure_initialized()
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name]


def list_datasets() -> list:
    """List all registered datasets."""
    _ensure_initialized()
    return list(_DATASET_REGISTRY.keys())
