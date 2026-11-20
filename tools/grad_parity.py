#!/usr/bin/env python3
"""grad_parity.py — P3 league gradient-direction parity diff.

Reads the Python gradient dump (py_keys.txt + py_grads.bin) and the C++ gradient dump
(cpp_keys.txt + cpp_grads.bin) for a fixture folder and asserts the per-parameter policy gradients
match: identical sorted key lists, identical flat length, cosine similarity > 0.99 and max|diff| small.

This is the P3 acceptance gate: the league routes each episode's experience to its owner's buffer and
runs the same PPO policy update; matching per-member gradients prove the routing's update math is correct.
"""
import argparse
import os
import sys

import numpy as np


def _read_keys(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _read_floats(path):
    return np.fromfile(path, dtype="<f4")


def main():
    ap = argparse.ArgumentParser(description="P3 gradient-direction parity diff")
    ap.add_argument("fixture_folder", help="folder with py_*/cpp_* key + grad dumps")
    ap.add_argument("--cos-min", type=float, default=0.99)
    ap.add_argument("--max-abs", type=float, default=1e-3)
    args = ap.parse_args()

    d = args.fixture_folder
    py_keys = _read_keys(os.path.join(d, "py_keys.txt"))
    cpp_keys = _read_keys(os.path.join(d, "cpp_keys.txt"))

    if py_keys != cpp_keys:
        print("KEY MISMATCH")
        print("  py-only :", sorted(set(py_keys) - set(cpp_keys))[:10])
        print("  cpp-only:", sorted(set(cpp_keys) - set(py_keys))[:10])
        return 1

    py = _read_floats(os.path.join(d, "py_grads.bin")).astype(np.float64)
    cpp = _read_floats(os.path.join(d, "cpp_grads.bin")).astype(np.float64)
    if py.shape != cpp.shape:
        print(f"SHAPE MISMATCH py={py.shape} cpp={cpp.shape}")
        return 1

    diff = np.abs(py - cpp)
    max_abs = float(diff.max())
    denom = float(np.linalg.norm(py) * np.linalg.norm(cpp))
    cos = float(np.dot(py, cpp) / denom) if denom > 0 else 0.0
    print(f"grad parity params={len(py_keys)} numel={py.size} "
          f"max|diff|={max_abs:.3e} cosine={cos:.8f}")

    if not (cos > args.cos_min):
        print(f"GRAD PARITY FAIL: cosine={cos:.8f} <= {args.cos_min}")
        return 1
    if not (max_abs < args.max_abs):
        print(f"GRAD PARITY FAIL: max|diff|={max_abs:.3e} >= {args.max_abs}")
        return 1
    print("GRAD PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
