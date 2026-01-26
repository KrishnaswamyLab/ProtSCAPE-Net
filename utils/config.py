"""
Config utility functions for loading and converting configuration files.
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Union, Dict, Any


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to config file (.yaml or .json)
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}. Use .yaml or .json")
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config


def config_to_hparams(config: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert config dictionary to argparse.Namespace object (hparams).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        argparse.Namespace containing hyperparameters
    """
    return argparse.Namespace(**config)


def save_config(config: Dict[str, Any], output_path: Union[str, Path], format: str = 'yaml') -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config file
        format: 'yaml' or 'json' (default: 'yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
