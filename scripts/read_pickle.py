#!/usr/bin/env python3
"""
Quick viewer for trajectory.pkl files saved by TrajectoryCollector.
Usage:
    python scripts/read_pickle.py /path/to/trajectory.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np


def describe_array(arr: Any) -> str:
    if isinstance(arr, np.ndarray):
        return f"ndarray(shape={arr.shape}, dtype={arr.dtype})"
    return f"{type(arr).__name__}"


def describe_observation(obs: dict) -> dict:
    summary = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            summary[k] = f"ndarray{v.shape} {v.dtype}"
        else:
            summary[k] = f"{type(v).__name__}"
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect trajectory.pkl content")
    parser.add_argument("pickle_path", type=Path, help="Path to trajectory.pkl")
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print each action row (index + values). Useful for inspecting trajectories.",
    )
    args = parser.parse_args()

    if not args.pickle_path.exists():
        raise SystemExit(f"File not found: {args.pickle_path}")

    with args.pickle_path.open("rb") as f:
        data = pickle.load(f)

    metadata = data.get("metadata", {})
    actions = data.get("actions")
    observations = data.get("observations", [])

    print("=== Metadata ===")
    pprint(metadata)
    print("\n=== Actions ===")
    if actions is None:
        print("None")
    else:
        print(describe_array(actions))
        if isinstance(actions, np.ndarray) and actions.size > 0:
            print("  first row sample:", actions[0])
            if args.print_actions:
                print("\n-- Actions (all rows) --")
                for idx, row in enumerate(actions):
                    print(f"[{idx:04d}] {row}")

    print("\n=== Observations ===")
    print(f"count: {len(observations)}")
    if observations:
        print("keys in first obs:", list(observations[0].keys()))
        print("summary of first obs fields:")
        pprint(describe_observation(observations[0]))


if __name__ == "__main__":
    main()
