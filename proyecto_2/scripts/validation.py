"""
Compare binary frame outputs between two model runs.

Default directories:
	frames/secuencial vs frames/simd
Usage:
  python validation.py [dir_a] [dir_b] [--atol 0.0] [--rtol 0.0]
"""

import argparse
import glob
import os
import sys

import numpy as np


def list_frames(frames_dir):
	pattern = os.path.join(frames_dir, "frame_*.bin")
	return sorted(glob.glob(pattern))


def load_frame(path):
	raw = np.fromfile(path, dtype=np.float32)
	if raw.size % 4 != 0:
		raise ValueError(f"Invalid frame size: {path}")
	return raw.reshape(-1, 4)


def report_mismatch(path, body_idx, comp_idx, a_val, b_val, max_diff=None):
	comp_labels = ["x", "y", "z", "gid"]
	comp = comp_labels[comp_idx] if comp_idx < len(comp_labels) else str(comp_idx)
	print(f"Mismatch in {path}")
	print(f"  body={body_idx} comp={comp} a={a_val} b={b_val}")
	if max_diff is not None:
		print(f"  max_abs_diff={max_diff}")


def compare_frames(dir_a, dir_b, atol, rtol):
	frames_a = list_frames(dir_a)
	frames_b = list_frames(dir_b)

	if not frames_a:
		print(f"Error: no frames found in {dir_a}/")
		return 2
	if not frames_b:
		print(f"Error: no frames found in {dir_b}/")
		return 2

	names_a = [os.path.basename(p) for p in frames_a]
	names_b = [os.path.basename(p) for p in frames_b]

	if names_a != names_b:
		set_a = set(names_a)
		set_b = set(names_b)
		missing_b = sorted(set_a - set_b)
		missing_a = sorted(set_b - set_a)
		print("Error: frame lists do not match")
		if missing_b:
			print(f"  missing in {dir_b}: {missing_b[:5]}")
		if missing_a:
			print(f"  missing in {dir_a}: {missing_a[:5]}")
		return 1

	strict = (atol == 0.0 and rtol == 0.0)

	for name in names_a:
		path_a = os.path.join(dir_a, name)
		path_b = os.path.join(dir_b, name)

		size_a = os.path.getsize(path_a)
		size_b = os.path.getsize(path_b)
		if size_a != size_b:
			print(f"Size mismatch in {name}: {size_a} vs {size_b}")
			return 1

		data_a = load_frame(path_a)
		data_b = load_frame(path_b)

		if data_a.shape != data_b.shape:
			print(f"Shape mismatch in {name}: {data_a.shape} vs {data_b.shape}")
			return 1

		if strict:
			if not np.array_equal(data_a, data_b):
				diff_idx = np.flatnonzero(data_a != data_b)
				flat = int(diff_idx[0])
				body = flat // 4
				comp = flat % 4
				report_mismatch(name, body, comp, data_a.flat[flat], data_b.flat[flat])
				return 1
		else:
			if not np.allclose(data_a, data_b, rtol=rtol, atol=atol):
				diff = np.abs(data_a - data_b)
				flat = int(np.argmax(diff))
				body = flat // 4
				comp = flat % 4
				report_mismatch(
					name,
					body,
					comp,
					data_a.flat[flat],
					data_b.flat[flat],
					float(diff.flat[flat]),
				)
				return 1

	print(f"OK: {len(names_a)} frames match")
	return 0


def main():
	parser = argparse.ArgumentParser(description="Compare frame outputs")
	parser.add_argument("dir_a", nargs="?", default="frames/secuencial")
	parser.add_argument("dir_b", nargs="?", default="frames/simd")
	parser.add_argument("--atol", type=float, default=1e-5)
	parser.add_argument("--rtol", type=float, default=1e-5)
	args = parser.parse_args()

	dir_a = args.dir_a.rstrip("/")
	dir_b = args.dir_b.rstrip("/")

	return compare_frames(dir_a, dir_b, args.atol, args.rtol)


if __name__ == "__main__":
	sys.exit(main())
