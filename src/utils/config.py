"""
Configuration management — load settings from config.yaml
"""
import os
import yaml
import torch
from pathlib import Path


class Config:
    """
    Load and access configuration from config.yaml

    Usage:
        config = Config()
        lr = config.generator.training.learning_rate
        seed = config.project.random_seed
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._set_attributes(self._config)

    def _load_config(self):
        """Load YAML config file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Make sure you're running from the project root directory."
            )
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _set_attributes(self, d, prefix=''):
        """Recursively set attributes from nested dict"""
        for key, value in d.items():
            if isinstance(value, dict):
                sub = ConfigSection(value)
                setattr(self, key, sub)
            else:
                setattr(self, key, value)

    def get_device(self):
        """Get the compute device (CPU or GPU)"""
        device_setting = self._config.get('project', {}).get('device', 'auto')
        if device_setting == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_setting)

    def __repr__(self):
        return f"Config(path='{self.config_path}')"


class ConfigSection:
    """A section of the configuration (for nested access)"""

    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items()
                 if not k.startswith('_')}
        return f"ConfigSection({attrs})"

    def to_dict(self):
        """Convert back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result