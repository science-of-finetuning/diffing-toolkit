#!/usr/bin/env python3
"""
Migrate on-disk results names for the diff mining rename.

This script renames result directories and updates per-run metadata:
- Method directory prefix:  logit_diff_topk_occurring_* -> diff_mining_*
- Extraction suffix:        _logit_extraction_direct    -> _logit_extraction_logits
- run_metadata.json:        logit_extraction_method: "direct" -> "logits"
                           and any string values starting with / equal to the old method key.

By default this runs in dry-run mode. Pass --apply to perform changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OLD_METHOD_KEY = "logit_diff_topk_occurring"
NEW_METHOD_KEY = "diff_mining"
OLD_EXTRACTION = "direct"
NEW_EXTRACTION = "logits"


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    dst: Path


def _rename_method_dir_name(name: str) -> str:
    assert name.startswith(OLD_METHOD_KEY), f"Expected {OLD_METHOD_KEY!r} prefix: {name!r}"
    out = NEW_METHOD_KEY + name[len(OLD_METHOD_KEY) :]
    out = out.replace("_logit_extraction_direct", "_logit_extraction_logits")
    return out


def _transform_json(obj: Any, *, key: str | None = None) -> Any:
    """
    Transform JSON values to reflect the rename.

    This is intentionally conservative:
    - Only exact matches or known prefixes/suffixes are rewritten.
    - Keys are left unchanged.
    """
    if isinstance(obj, dict):
        return {k: _transform_json(v, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_transform_json(v, key=key) for v in obj]
    if isinstance(obj, str):
        if obj == OLD_METHOD_KEY:
            return NEW_METHOD_KEY
        if key == "logit_extraction_method" and obj == OLD_EXTRACTION:
            return NEW_EXTRACTION
        if obj.startswith(OLD_METHOD_KEY):
            return _rename_method_dir_name(obj)
        if "_logit_extraction_direct" in obj:
            return obj.replace("_logit_extraction_direct", "_logit_extraction_logits")
        return obj
    return obj


def _iter_method_dirs(results_base_dir: Path) -> Iterable[Path]:
    """
    Yield method directories under the expected layout:
      results_base_dir / {model} / {organism} / {method_dir}
    """
    for model_dir in results_base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for organism_dir in model_dir.iterdir():
            if not organism_dir.is_dir():
                continue
            for method_dir in organism_dir.iterdir():
                if method_dir.is_dir() and method_dir.name.startswith(OLD_METHOD_KEY):
                    yield method_dir


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _migrate_run_metadata(method_dir: Path, *, apply: bool) -> None:
    for meta_path in sorted(method_dir.glob("run_*/run_metadata.json")):
        before = _load_json(meta_path)
        after = _transform_json(before)
        if before == after:
            continue

        print(f"- update {meta_path}")
        if apply:
            _write_json(meta_path, after)


def _plan_renames(results_base_dirs: list[Path]) -> list[RenamePlan]:
    plans: list[RenamePlan] = []
    for base in results_base_dirs:
        assert base.exists(), f"--results-base-dir does not exist: {base}"
        assert base.is_dir(), f"--results-base-dir is not a directory: {base}"
        for method_dir in _iter_method_dirs(base):
            new_name = _rename_method_dir_name(method_dir.name)
            dst = method_dir.with_name(new_name)
            plans.append(RenamePlan(src=method_dir, dst=dst))

    # Deterministic order, shallow first is fine (method dirs should be leaves).
    plans.sort(key=lambda p: str(p.src))
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate existing results from logit_diff_topk_occurring -> diff_mining."
    )
    parser.add_argument(
        "--results-base-dir",
        action="append",
        type=Path,
        required=True,
        help="Root results dir (e.g. .../diffing_results). Can be passed multiple times.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename directories and rewrite metadata (default: dry-run).",
    )
    args = parser.parse_args()

    results_base_dirs = [p.expanduser() for p in args.results_base_dir]
    apply = bool(args.apply)

    plans = _plan_renames(results_base_dirs)
    if not plans:
        print("No directories to rename.")
        return

    print(f"Planned renames ({len(plans)}):")
    for p in plans:
        print(f"- {p.src} -> {p.dst}")

    # Collision check (fail fast).
    for p in plans:
        if p.dst.exists():
            raise RuntimeError(f"Destination already exists: {p.dst}")

    if not apply:
        print("\nDry-run mode (no changes made). Re-run with --apply to perform the migration.")
        return

    for p in plans:
        print(f"\nRenaming directory:\n- {p.src}\n- {p.dst}")
        p.src.rename(p.dst)
        _migrate_run_metadata(p.dst, apply=True)


if __name__ == "__main__":
    main()

