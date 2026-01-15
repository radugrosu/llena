from pathlib import Path
import re

import os
from typing import Any, Iterable
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_vars(value: str) -> str:
    """
    Expand patterns like:
      "${VAR}" -> os.environ.get("VAR", "")
      "${VAR:-default}" -> os.environ.get("VAR", "default")
    """

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(key, default)

    return _ENV_PATTERN.sub(repl, value)


def _walk_and_expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Merge override into base (recursively), returning a new dict.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted_key(cfg: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """
    Set nested keys like "train.lr_lora=1e-4" into cfg.

    Attempts type coercion:
      - "true"/"false" -> bool
      - "null"/"none" -> None
      - ints/floats
      - YAML literals (lists/dicts) if provided (e.g. "[1,2]" or "{a:1}")
      - otherwise string
    """
    # Try to parse as YAML scalar/structure first (handles numbers, bools, lists, dicts)
    try:
        value = yaml.safe_load(raw_value)
    except Exception:
        value = raw_value

    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_config(
    config_path: str | Path,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Load YAML config, optionally support a base config via top-level key:
      base_config: "configs/base.yaml"

    Apply environment variable expansion and dotted-key CLI overrides:
      overrides=["train.lr_lora=5e-5", "mm.num_image_tokens=128"]
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Optional base config chaining
    base_ref = cfg.pop("base_config", None)
    if base_ref:
        base_path = (config_path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)  # recursive; base can also have base_config
        cfg = _deep_merge(base_cfg, cfg)

    # Expand env vars like ${HF_HOME:-...}
    cfg = _walk_and_expand(cfg)

    # Apply overrides
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be KEY=VALUE, got: {item}")
            k, v = item.split("=", 1)
            _set_dotted_key(cfg, k.strip(), v.strip())

    # Minimal required keys check (add more later if you want)
    required = [
        ("model", "llm_name"),
        ("model", "vision_name"),
        ("mm", "num_image_tokens"),
        ("train", "max_seq_len"),
        ("data", "dataset"),
    ]
    for a, b in required:
        if a not in cfg or not isinstance(cfg[a], dict) or b not in cfg[a]:
            raise KeyError(f"Missing required config key: {a}.{b}")

    return cfg


def save_resolved_config(cfg: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
