#!/usr/bin/env python3
"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data_test_3dhp.npz"


def convert_test(test_root, output_path):
    test_root = Path(test_root)
    output_path = Path(output_path)
    data_by_subject = {}

    for annot_path in sorted(test_root.glob("TS*/annot_data.mat")):
        subject = annot_path.parent.name
        print(f"loading {subject}...")

        with h5py.File(annot_path, "r") as data:
            valid_frame = np.squeeze(data["valid_frame"][()])
            data_2d = np.squeeze(data["annot2"][()])
            data_3d = np.squeeze(data["univ_annot3"][()])

        data_by_subject[subject] = {
            "data_2d": data_2d,
            "data_3d": data_3d,
            "valid": valid_frame,
        }

    if not data_by_subject:
        raise FileNotFoundError(f"No test annot_data.mat files found under {test_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, data=data_by_subject)
    print(f"saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MPI-INF-3DHP test annotations to the FMPose3D 3DHP test npz."
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        required=True,
        help="Path containing TS1..TS6 folders from the official MPI-INF-3DHP test set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output npz path consumed by 3dhp_test/infer_3dhp.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_test(args.test_root, args.output)
