from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import tomllib


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix} (use .toml)")


def _coerce_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, list):
        expects_list = action.nargs in {"+", "*"} or isinstance(action.nargs, int)
        if expects_list:
            if action.type is None:
                return value
            return [action.type(v) for v in value]
        if len(value) == 1:
            value = value[0]
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
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Path to TOML run config")
    return parser.parse_args(argv)
