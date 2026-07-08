"""
Loads YAML configs with environment-specific overrides.
Usage:
    from drivemind.common.config_loader import load_config
    cfg = load_config("dev")
"""

import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str = "dev") -> Dict[str, Any]:
    base_path = CONFIG_DIR / "base.yaml"
    env_path = CONFIG_DIR / f"{env}.yaml"

    base_cfg = {}
    if base_path.exists() and base_path.stat().st_size > 0:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f) or {}

    env_cfg = {}
    if env_path.exists() and env_path.stat().st_size > 0:
        with open(env_path, "r") as f:
            env_cfg = yaml.safe_load(f) or {}

    return _deep_merge(base_cfg, env_cfg)