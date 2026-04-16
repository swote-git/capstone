from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import tomllib


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "YAML config requires PyYAML. Install with `pip install pyyaml`."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return data
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def _coerce_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return value
    if action.nargs in {"+", "*"} and isinstance(value, list):
        if action.type is None:
            return value
        return [action.type(v) for v in value]
    if action.type is not None and not isinstance(value, bool):
        return action.type(value)
    return value


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    defaults: Dict[str, Any] = {}
    if pre_args.config is not None:
        raw = load_config(pre_args.config)
        common = raw.get("common", {})
        scoped = raw.get(section, {})
        if isinstance(common, dict):
            defaults.update(common)
        if isinstance(scoped, dict):
            defaults.update(scoped)
        defaults = {k.replace("-", "_"): v for k, v in defaults.items()}

        action_map = {a.dest: a for a in parser._actions if a.dest != "help"}
        for k, v in list(defaults.items()):
            if k in action_map:
                defaults[k] = _coerce_value(action_map[k], v)

    parser.set_defaults(**defaults)
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Path to TOML/YAML/JSON run config")
    return parser.parse_args(argv)
