#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Determine which UT tests to run based on changed files in a PR.

This script reads ut_config.yaml to understand the mapping between source
directories and their corresponding UT test directories.  Given a list of
changed files (from git diff), it identifies which modules are affected,
scans test files for ``@npu_test`` decorators, and groups tests by
(num_npus, npu_type) so the workflow can route them to the correct runner.

Routing rules:
  - Tests decorated with ``@npu_test(num_npus=N, npu_type=T)`` are routed
    to the runner that exactly matches (T, N) in runner_label.json.
  - Tests **without** ``@npu_test`` are routed to the CPU runner.
  - For class-based tests the decorator goes on the **class**, and the
    output is the class-level node ID (``file::Class``).

Output granularity:
  - Directory whose files are ALL undecorated → directory path.
  - Mixed directory (some files have ``@npu_test``) →
      decorated files: function / class node IDs per runner;
      undecorated files: file paths in the CPU group.

Usage:
    python determine_smart_e2e_scope.py --changed-files file1.py file2.py
    python determine_smart_e2e_scope.py --diff-base origin/main

Output (written to $GITHUB_OUTPUT if available, otherwise stdout):
    test_groups=<JSON array of {num_npus, npu_type, runner, tests}>
    has_tests=true/false
    matched_modules=<comma-separated module names>
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants & types
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_CONFIG_PATH = _SCRIPT_DIR / "ut_config.yaml"
_RUNNER_LABEL_PATH = _SCRIPT_DIR / "runner_label.json"


class RunnerDeviceType(str, Enum):
    """Chip types — values must match runner_label.json ``chip`` field exactly.

    Shared by:
      - tests/ut/conftest.py  (``npu_test`` decorator)
      - .github/workflows/scripts/determine_smart_e2e_scope.py  (AST parser)
    """

    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    CPU = "cpu"


# Type alias for the key used to group tests by runner requirements.
RunnerKey = tuple[int, RunnerDeviceType]

# Tests without @npu_test decorator route to the CPU runner.
_DEFAULT_KEY: RunnerKey = (0, RunnerDeviceType.CPU)


@dataclass
class RunnerInfo:
    """A self-hosted runner entry parsed from runner_label.json."""

    num_npus: int
    npu_type: RunnerDeviceType
    label: str
    image_tag: str = ""


# ---------------------------------------------------------------------------
# Runner resolution
# ---------------------------------------------------------------------------


def load_runners() -> list[RunnerInfo]:
    """Load runner definitions from runner_label.json."""
    with open(_RUNNER_LABEL_PATH) as f:
        raw = json.load(f)
    return [
        RunnerInfo(
            num_npus=info["npu_num"],
            npu_type=RunnerDeviceType(info["chip"]),
            label=label,
            image_tag=info.get("image_tag", ""),
        )
        for label, info in raw.items()
    ]


def resolve_runner(
    num_npus: int,
    npu_type: RunnerDeviceType,
    runners: list[RunnerInfo],
) -> RunnerInfo | None:
    """Find the exact matching runner for the given NPU requirements.

    For NPU types: exact match on (npu_type, num_npus).
    For CPU type:  match any CPU runner.

    Returns None if no matching runner exists.
    """
    if npu_type == RunnerDeviceType.CPU:
        candidates = [r for r in runners if r.npu_type == RunnerDeviceType.CPU]
    else:
        candidates = [r for r in runners if r.npu_type == npu_type and r.num_npus == num_npus]
    if not candidates:
        return None
    return candidates[0]


# ---------------------------------------------------------------------------
# Git & module matching
# ---------------------------------------------------------------------------


def get_changed_files(base_ref: str) -> list[str]:
    """Return the list of changed files by diffing against *base_ref*."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def _any_file_matches(changed_files: list[str], deps: list[str]) -> bool:
    """Return True if any changed file falls under any dependency prefix."""
    for changed in changed_files:
        for dep in deps:
            if changed == dep or changed.startswith(dep + "/"):
                return True
    return False


def match_modules(
    changed_files: list[str],
    config: list[dict],
) -> list[str]:
    """Return names of modules whose source dependencies match changed files.

    - ``optional: false`` → always included when there are changes.
    - ``optional: true``  → included only when a changed file falls under
      one of the module's ``source_file_dependencies``.
    """
    matched: list[str] = []
    if not changed_files:
        return matched

    for module in config:
        if not module.get("optional", True):
            matched.append(module["name"])
            continue
        deps = module.get("source_file_dependencies", [])
        if _any_file_matches(changed_files, deps):
            matched.append(module["name"])
    return matched


def collect_test_dirs(
    matched_modules: list[str],
    config: list[dict],
) -> list[str]:
    """Collect deduplicated test directory paths for matched modules."""
    module_map = {m["name"]: m for m in config}
    test_dirs: set[str] = set()
    for name in matched_modules:
        for path in module_map[name].get("tests", []):
            test_dirs.add(path)
    return sorted(test_dirs)


# ---------------------------------------------------------------------------
# AST scanning: extract @npu_test decorator info from test files
# ---------------------------------------------------------------------------


def _resolve_npu_type_from_ast(node: ast.expr) -> RunnerDeviceType:
    """Resolve an AST node to a RunnerDeviceType value.

    Handles:
      - String literal:   ``npu_type="a3"``               → ast.Constant
      - Enum attribute:   ``npu_type=RunnerDeviceType.A3`` → ast.Attribute
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return RunnerDeviceType(node.value)
    if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
        return RunnerDeviceType[node.attr]
    raise ValueError(f"Cannot resolve npu_type from AST node: {ast.dump(node)}")


def _extract_runner_key(
    node: ast.FunctionDef | ast.ClassDef,
) -> RunnerKey:
    """Extract (num_npus, npu_type) from ``@npu_test`` on *node*.

    Returns ``_DEFAULT_KEY`` when no ``@npu_test`` decorator is found.
    Default values (``num_npus=1, npu_type=A2``) match
    ``tests/ut/conftest.py::npu_test``.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        decorator_func = decorator.func
        if not (isinstance(decorator_func, ast.Name) and decorator_func.id == "npu_test"):
            continue

        num_npus = 1
        npu_type = RunnerDeviceType.A2
        for kw in decorator.keywords:
            if kw.arg == "num_npus" and isinstance(kw.value, ast.Constant):
                num_npus = kw.value.value
            elif kw.arg == "npu_type":
                npu_type = _resolve_npu_type_from_ast(kw.value)
        return (num_npus, npu_type)

    return _DEFAULT_KEY


def scan_test_file(filepath: str) -> dict[RunnerKey, list[str]]:
    """Parse a single test file and group node IDs by runner key.

    - Top-level ``test_*`` functions → keyed by their own ``@npu_test``.
    - Classes → keyed by the **class-level** ``@npu_test``; output the
      class node ID (``file::Class``), not individual methods.
    - Nodes without ``@npu_test`` → keyed by ``_DEFAULT_KEY`` (CPU).
    """
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    groups: dict[RunnerKey, list[str]] = defaultdict(list)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") or isinstance(node, ast.ClassDef):
            key = _extract_runner_key(node)
            groups[key].append(f"{filepath}::{node.name}")

    return dict(groups)


# ---------------------------------------------------------------------------
# Grouping & runner resolution
# ---------------------------------------------------------------------------


def group_tests_by_runner(
    test_dirs: list[str],
    runners: list[RunnerInfo],
) -> list[dict]:
    """Scan test directories and group tests by runner requirements.

    Output granularity depends on whether a directory contains ``@npu_test``:

    - **Pure-default directory** (no ``@npu_test`` anywhere):
          directory path → CPU group.
    - **Mixed directory** (some files have ``@npu_test``):
          decorated files → function / class node IDs per runner;
          undecorated files → file paths in CPU group.

    Exits with an error if any (num_npus, npu_type) combination cannot be
    matched to a runner in runner_label.json.
    """
    all_groups: dict[RunnerKey, list[str]] = defaultdict(list)

    for test_dir in test_dirs:
        _scan_directory(test_dir, all_groups)

    return _resolve_groups(all_groups, runners)


def _scan_directory(
    test_dir: str,
    all_groups: dict[RunnerKey, list[str]],
) -> None:
    """Scan a single test directory and populate *all_groups* in place."""
    dir_path = Path(test_dir)
    if not dir_path.exists():
        all_groups[_DEFAULT_KEY].append(test_dir)
        return

    has_decorated_tests = False
    undecorated_files: list[str] = []

    for test_file in sorted(dir_path.rglob("test_*.py")):
        file_path = str(test_file)
        file_groups = scan_test_file(file_path)

        if not file_groups:
            undecorated_files.append(file_path)
            continue

        if any(key != _DEFAULT_KEY for key in file_groups):
            has_decorated_tests = True
            for key, node_ids in file_groups.items():
                all_groups[key].extend(node_ids)
        else:
            undecorated_files.append(file_path)

    if not has_decorated_tests:
        # Entire directory is undecorated → single directory path
        all_groups[_DEFAULT_KEY].append(test_dir)
    else:
        # Mixed directory → keep undecorated files as file-level paths
        all_groups[_DEFAULT_KEY].extend(undecorated_files)


def _resolve_groups(
    all_groups: dict[RunnerKey, list[str]],
    runners: list[RunnerInfo],
) -> list[dict]:
    """Convert internal groups to output dicts with runner labels.

    Exits with a descriptive error if any group cannot be resolved.
    """
    result: list[dict] = []
    errors: list[str] = []

    for (num_npus, npu_type), tests in sorted(all_groups.items()):
        if not tests:
            continue

        runner_info = resolve_runner(num_npus, npu_type, runners)
        if runner_info is None:
            errors.append(_format_runner_error(num_npus, npu_type, tests, runners))
            continue

        group = {
            "num_npus": num_npus,
            "npu_type": npu_type.value,
            "runner": runner_info.label,
            "tests": " ".join(sorted(set(tests))),
        }
        if runner_info.image_tag:
            group["image_tag"] = runner_info.image_tag
        result.append(group)

    if errors:
        print(
            "\nERROR: The following @npu_test decorator combinations "
            "cannot be routed to any runner in runner_label.json:"
            + "".join(errors)
            + "\n\nPlease fix the decorator arguments or add the "
            "missing runner to runner_label.json.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return result


def _format_runner_error(
    num_npus: int,
    npu_type: RunnerDeviceType,
    tests: list[str],
    runners: list[RunnerInfo],
) -> str:
    """Build a human-readable error message for an unresolvable group."""
    available = [f"{r.label} ({r.npu_type.value} x{r.num_npus})" for r in runners if r.npu_type == npu_type]
    header = f'\n  @npu_test(num_npus={num_npus}, npu_type="{npu_type.value}") — no runner available.'
    runners_line = (
        f"\n    Available {npu_type.value} runners: {', '.join(available)}"
        if available
        else f'\n    No runners defined for chip type "{npu_type.value}".'
    )
    tests_line = "\n    Affected tests:\n" + "\n".join(f"      - {t}" for t in sorted(set(tests)))
    return header + runners_line + tests_line


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_output(
    test_groups: list[dict],
    matched_modules: list[str],
) -> None:
    """Write step outputs to $GITHUB_OUTPUT (or stdout when running locally)."""
    has_tests = len(test_groups) > 0
    groups_json = json.dumps(test_groups, separators=(",", ":"))

    outputs = {
        "test_groups": groups_json,
        "has_tests": str(has_tests).lower(),
        "matched_modules": ",".join(matched_modules),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")

    _print_summary(test_groups, matched_modules, has_tests)


def _print_summary(
    test_groups: list[dict],
    matched_modules: list[str],
    has_tests: bool,
) -> None:
    """Print a human-readable summary to stderr for CI logs."""
    divider = "=" * 60
    print(f"\n{divider}", file=sys.stderr)
    print("Smart UT Scope Determination Summary", file=sys.stderr)
    print(divider, file=sys.stderr)
    print(f"Matched modules: {matched_modules or '(none)'}", file=sys.stderr)
    print(f"Has tests to run: {has_tests}", file=sys.stderr)

    for group in test_groups:
        npu_type = group["npu_type"]
        num_npus = group["num_npus"]
        runner = group["runner"]
        tests = group["tests"].split()
        print(
            f"\n  [{npu_type} x{num_npus}] -> {runner} ({len(tests)} tests):",
            file=sys.stderr,
        )
        for t in tests:
            print(f"    - {t}", file=sys.stderr)

    print(f"{divider}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Determine UT test scope based on changed files",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--changed-files",
        nargs="+",
        help="List of changed file paths",
    )
    input_group.add_argument(
        "--diff-base",
        type=str,
        help="Git ref to diff against (e.g. origin/main)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to ut_config.yaml",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    changed_files = get_changed_files(args.diff_base) if args.diff_base else args.changed_files

    matched_modules = match_modules(changed_files, config)
    test_dirs = collect_test_dirs(matched_modules, config)
    runners = load_runners()
    test_groups = group_tests_by_runner(test_dirs, runners)
    write_output(test_groups, matched_modules)


if __name__ == "__main__":
    main()
