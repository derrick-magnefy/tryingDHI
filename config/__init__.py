"""
config - Configuration Management

Manages configuration for PD analysis:
- Default feature selections
- Classification thresholds
- Clustering parameters
- ADC settings

Configuration files are in YAML format under config/defaults/.
Local overrides can be placed in config/local/ (gitignored).

Usage:
    from config import get_config, get_thresholds, get_features

    # Get specific configurations
    thresholds = get_thresholds()
    features = get_features()

    # Or use the loader directly
    config = get_config()
    clustering = config.get_clustering()
"""

from config.loader import (
    ConfigLoader,
    get_config,
    get_thresholds,
    get_features,
    get_clustering,
    get_settings,
)

__all__ = [
    'ConfigLoader',
    'get_config',
    'get_thresholds',
    'get_features',
    'get_clustering',
    'get_settings',
]
