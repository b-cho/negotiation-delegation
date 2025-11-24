"""Configuration loader for YAML files"""
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model configuration from main config
    
    Args:
        config: Main configuration dictionary
    
    Returns:
        Model configuration dictionary
    """
    return config.get("model", {})


def get_experiment_config(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """
    Extract experiment-specific configuration
    
    Args:
        config: Main configuration dictionary
        experiment_name: Name of the experiment ('experiment1' or 'experiment2')
    
    Returns:
        Experiment configuration dictionary
    """
    experiments = config.get("experiments", {})
    return experiments.get(experiment_name, {})


def get_house_specs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract house specifications from config
    
    Args:
        config: Main configuration dictionary
    
    Returns:
        House specifications dictionary
    """
    return config.get("house_specs", {})


def get_profiles(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract buyer/seller profile definitions from config
    
    Args:
        config: Main configuration dictionary
    
    Returns:
        Profiles configuration dictionary
    """
    return config.get("profiles", {})


def get_statistical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract statistical analysis configuration
    
    Args:
        config: Main configuration dictionary
    
    Returns:
        Statistical configuration dictionary
    """
    return config.get("statistics", {})

