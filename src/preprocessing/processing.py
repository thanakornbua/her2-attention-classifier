"""Stain-normalization utilities for streaming image data into Zarr.

This module provides helpers to normalize RGB images using precomputed
reference stain vectors (e.g. Macenko vectors) and to efficiently write
the normalized images into Zarr stores. Functions are written to be
memory-conscious and can optionally leverage GPU acceleration via CuPy.

All public functions and helpers follow PEP 257 docstring conventions.
"""

import numpy as np
import cv2
import zarr
from pathlib import Path


def _get_array_lib(use_gpu: bool):
	"""Return cupy if available and requested, otherwise numpy."""
	if use_gpu:
		try:
			import cupy

			return cupy
		except ImportError:
			pass
	return np


def _normalize_with_reference(image: np.ndarray, reference_stain_vectors, xp):
	"""Apply reference stain vectors using OD reconstruction (GPU or CPU)."""
	img = xp.asarray(image, dtype=xp.float32)
	od = -xp.log((img + 1.0) / 256.0)

	ref = xp.asarray(reference_stain_vectors, dtype=xp.float32)  # (2, 3)
	od_flat = od.reshape(-1, 3)

	# Least squares to get concentrations in reference basis
	conc = xp.linalg.lstsq(ref.T, od_flat.T, rcond=None)[0].T  # (N, 2)

	# Recompose with reference vectors
	od_norm = conc @ ref  # (N, 3)
	rgb = xp.exp(-od_norm) * 256.0
	rgb = xp.clip(rgb, 0, 255).reshape(image.shape).astype(xp.uint8)
	return rgb


def normalize_images_to_zarr(
	image_paths,
	reference_stain_vectors,
	zarr_path,
	resize_shape=None,
	use_gpu: bool = True,
	chunk_len: int = 1,
):
	"""
	Normalize a sequence of RGB images with reference stain vectors and stream to Zarr.

	Args:
		image_paths (Sequence[str]): Paths to input RGB images (any OpenCV-readable format).
		reference_stain_vectors (array-like): Reference stain matrix with shape (2, 3).
		zarr_path (str | Path): Output Zarr store path.
		resize_shape (tuple[int, int], optional): (height, width) to resize images.
		use_gpu (bool, optional): Use CuPy/CUDA if available. Defaults to True.
		chunk_len (int, optional): Chunk length along the image axis for Zarr. Defaults to 1.

	Notes:
		- Memory efficient: processes images one-by-one; no full batch kept in RAM/VRAM.
		- Speed: uses CuPy when available; falls back to NumPy otherwise.
		- Assumes all images are (or resized to) the same spatial shape.
	"""

	xp = _get_array_lib(use_gpu)

	paths = list(image_paths)
	if len(paths) == 0:
		raise ValueError("No images provided for normalization")

	# Peek first image to define dataset shape
	first = cv2.cvtColor(cv2.imread(paths[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
	if resize_shape is not None:
		first = cv2.resize(first, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_AREA)

	height, width, _ = first.shape
	store_path = Path(zarr_path)
	store_path.parent.mkdir(parents=True, exist_ok=True)

	# Create Zarr array
	z = zarr.open(
		store_path,
		mode="w",
		shape=(len(paths), height, width, 3),
		chunks=(chunk_len, height, width, 3),
		dtype="uint8",
		compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE),
	)

	# Process first image
	norm_first = _normalize_with_reference(first, reference_stain_vectors, xp)
	z[0] = norm_first.get() if hasattr(norm_first, "get") else norm_first

	# Process remaining images
	for idx, p in enumerate(paths[1:], start=1):
		img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		if resize_shape is not None:
			img = cv2.resize(img, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_AREA)

		norm = _normalize_with_reference(img, reference_stain_vectors, xp)
		z[idx] = norm.get() if hasattr(norm, "get") else norm

	return store_path
