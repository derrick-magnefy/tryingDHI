"""
Configuration Loader

Loads configuration from YAML files with support for:
- Default configurations in config/defaults/
- Local overrides in config/local/ (gitignored for development)
- Runtime overrides passed as dictionaries
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """
    Load and manage configuration from YAML files.

    Usage:
        config = ConfigLoader()
        thresholds = config.get_thresholds()
        features = config.get_features()
        clustering = config.get_clustering()
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Base config directory. Defaults to this package's directory.
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path(__file__).parent

        self.defaults_dir = self.config_dir / 'defaults'
        self.local_dir = self.config_dir / 'local'
        self._cache: Dict[str, Any] = {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries, with override taking precedence.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load(self, name: str, local_overrides: Optional[Dict] = None, use_cache: bool = True) -> Dict:
        """
        Load a configuration file with optional overrides.

        Args:
            name: Config file name without extension (e.g., 'thresholds')
            local_overrides: Runtime overrides to apply
            use_cache: Whether to use cached values

        Returns:
            Dict containing the merged configuration
        """
        cache_key = name

        # Return cached value if available and no overrides
        if use_cache and cache_key in self._cache and local_overrides is None:
            return self._cache[cache_key]

        # Load default config
        default_path = self.defaults_dir / f'{name}.yaml'
        if not default_path.exists():
            raise FileNotFoundError(f"Default config not found: {default_path}")

        with open(default_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Load local override if exists
        local_path = self.local_dir / f'{name}.yaml'
        if local_path.exists():
            with open(local_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
            config = self._deep_merge(config, local_config)

        # Apply runtime overrides
        if local_overrides:
            config = self._deep_merge(config, local_overrides)

        # Cache the result (without runtime overrides)
        if local_overrides is None:
            self._cache[cache_key] = config

        return config

    def get_thresholds(self, overrides: Optional[Dict] = None) -> Dict:
        """Load classification thresholds configuration."""
        return self.load('thresholds', overrides)

    def get_features(self, overrides: Optional[Dict] = None) -> Dict:
        """Load feature configuration."""
        return self.load('features', overrides)

    def get_clustering(self, overrides: Optional[Dict] = None) -> Dict:
        """Load clustering configuration."""
        return self.load('clustering', overrides)

    def get_settings(self, overrides: Optional[Dict] = None) -> Dict:
        """Load general settings configuration."""
        return self.load('settings', overrides)

    def get_default_clustering_features(self) -> list:
        """Get the default features for clustering."""
        features = self.get_features()
        return features.get('pulse_features', {}).get('default_clustering', [])

    def get_default_feature_weights(self) -> Dict[str, float]:
        """Get the default feature weights for clustering."""
        features = self.get_features()
        return features.get('pulse_features', {}).get('default_weights', {})

    def get_noise_thresholds(self) -> Dict:
        """Get noise detection thresholds."""
        thresholds = self.get_thresholds()
        return thresholds.get('noise_detection', {})

    def get_surface_thresholds(self) -> Dict:
        """Get surface PD detection thresholds."""
        thresholds = self.get_thresholds()
        return thresholds.get('surface_detection', {})

    def get_corona_internal_thresholds(self) -> Dict:
        """Get corona/internal detection thresholds."""
        thresholds = self.get_thresholds()
        return thresholds.get('corona_internal', {})

    def save_local(self, name: str, config: Dict) -> Path:
        """
        Save configuration to local override file.

        Args:
            name: Config file name without extension
            config: Configuration to save

        Returns:
            Path to saved file
        """
        # Ensure local directory exists
        self.local_dir.mkdir(parents=True, exist_ok=True)

        local_path = self.local_dir / f'{name}.yaml'
        with open(local_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Clear cache for this config
        if name in self._cache:
            del self._cache[name]

        return local_path

    def clear_cache(self):
        """Clear all cached configurations."""
        self._cache.clear()


# Singleton instance for convenience
_default_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get the default config loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def get_thresholds(overrides: Optional[Dict] = None) -> Dict:
    """Convenience function to get thresholds."""
    return get_config().get_thresholds(overrides)


def get_features(overrides: Optional[Dict] = None) -> Dict:
    """Convenience function to get features config."""
    return get_config().get_features(overrides)


def get_clustering(overrides: Optional[Dict] = None) -> Dict:
    """Convenience function to get clustering config."""
    return get_config().get_clustering(overrides)


def get_settings(overrides: Optional[Dict] = None) -> Dict:
    """Convenience function to get settings."""
    return get_config().get_settings(overrides)
