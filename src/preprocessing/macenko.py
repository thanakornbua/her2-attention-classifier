"""
Macenko Stain Normalization

Implements the Macenko stain normalization method for histopathological images.
This method estimates stain vectors from histopathological images and normalizes
them to a reference stain vector, making images comparable across different
staining batches and microscopes.

References:
    Macenko, M., Niethammer, M., Marron, J. S., Borland, D., Woosley, J. T.,
    Guan, X., ... & Thomas, N. E. (2009). "A method for normalizing histology
    slides for quantitative analysis." IEEE International Symposium on
    Biomedical Imaging (ISBI).
"""

import numpy as np
from typing import Tuple, Optional, Union


def get_macenko_vectors(
    rgb_image: np.ndarray,
    luminosity_threshold: int = 10,
    angular_percentile: float = 99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate stain vectors from a histopathological RGB image using Macenko method.

    This function estimates the two primary stain vectors (typically H&E - Hematoxylin
    and Eosin) from a histology image by analyzing the optical density distribution.

    Args:
        rgb_image (np.ndarray): RGB image as numpy array with shape (height, width, 3).
            Values should be in range [0, 255] with dtype uint8.
        luminosity_threshold (int, optional): Pixels with luminosity below this value
            are considered background and excluded. Defaults to 10.
        angular_percentile (float, optional): Percentile for selecting extreme angles
            in the angular distribution. Defaults to 99 (typically 99th or 95th percentile).

    Returns:
        tuple: (stain_vector_1, stain_vector_2) - Two normalized stain vectors
            Each vector has shape (3,) and represents the direction of one stain in
            optical density space (log-transformed RGB).

    Raises:
        ValueError: If input image shape is invalid.
        ValueError: If no foreground pixels found (image too bright/white).

    Example:
        >>> patch = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> stain1, stain2 = get_macenko_vectors(patch)
        >>> print(stain1.shape, stain2.shape)  # (3,) (3,)
    """
    # Input validation
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")

    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)

    # Flatten image to pixels
    h, w, c = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)

    # Convert RGB to optical density (OD)
    # Add small epsilon to avoid log(0)
    od = -np.log((pixels + 1) / 256.0)

    # Calculate luminosity (mean across color channels)
    luminosity = np.mean(od, axis=1)

    # Mask out background pixels (high luminosity = white background)
    mask = luminosity > (luminosity_threshold / 255.0)
    od_foreground = od[mask]

    if len(od_foreground) == 0:
        raise ValueError(
            "No foreground pixels found. Image may be too bright or threshold too high. "
            f"Consider lowering luminosity_threshold (current: {luminosity_threshold})"
        )

    # Normalize by luminosity
    od_normalized = od_foreground / np.reshape(
        np.linalg.norm(od_foreground, axis=1), (-1, 1)
    )

    # Find extreme angles in the distribution
    # Use SVD to find directions of maximum variance
    # Note: SVD of (N, 3) matrix returns U(N,3), s(3,), Vt(3,3)
    # We want the right singular vectors (columns of V), not left (rows of U)
    _, _, Vt = np.linalg.svd(od_normalized, full_matrices=False)
    V = Vt.T

    # Get the two primary directions (first two columns of V)
    # These correspond to the two main stain directions
    stain_vec_1 = V[:, 0]
    stain_vec_2 = V[:, 1]

    # Ensure vectors are in the correct direction (positive in first component)
    if stain_vec_1[0] < 0:
        stain_vec_1 = -stain_vec_1
    if stain_vec_2[0] < 0:
        stain_vec_2 = -stain_vec_2

    return stain_vec_1, stain_vec_2


def get_macenko_concentrations(
    rgb_image: np.ndarray,
    stain_vectors: np.ndarray,
    luminosity_threshold: int = 10
) -> np.ndarray:
    """
    Calculate stain concentrations for a given image and stain vectors.

    Args:
        rgb_image (np.ndarray): RGB image with shape (height, width, 3).
        stain_vectors (np.ndarray): Stain vectors with shape (2, 3), where each row
            is a normalized stain vector.
        luminosity_threshold (int, optional): Background threshold. Defaults to 10.

    Returns:
        np.ndarray: Stain concentrations with shape (height*width, 2).
            Each row contains the concentrations of the two stains for that pixel.

    Example:
        >>> patch = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> stain_vecs = np.array([[0.5, 0.6, 0.7], [0.8, 0.2, 0.1]])
        >>> concentrations = get_macenko_concentrations(patch, stain_vecs)
        >>> print(concentrations.shape)  # (65536, 2)
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)

    h, w, c = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)

    # Convert to optical density
    od = -np.log((pixels + 1) / 256.0)

    # Solve least squares problem: OD = C * stain_vectors.T
    # where C are the concentrations
    concentrations = np.linalg.lstsq(stain_vectors.T, od.T, rcond=None)[0].T

    return concentrations


def macenko_stain_normalize(
    rgb_image: np.ndarray,
    reference_stain_vectors: Optional[np.ndarray] = None,
    luminosity_threshold: int = 10,
    angular_percentile: float = 99,
    concentration_clip: Tuple[float, float] = (0.0, np.inf)
) -> np.ndarray:
    """
    Apply Macenko stain normalization to a histopathological image.

    Normalizes an image to match reference stain vectors. If no reference is provided,
    uses the image's own stain vectors (useful for batch processing consistency).

    Args:
        rgb_image (np.ndarray): RGB image with shape (height, width, 3) in range [0, 255].
        reference_stain_vectors (np.ndarray, optional): Reference stain vectors with shape (2, 3).
            If None, uses stain vectors estimated from the input image.
        luminosity_threshold (int, optional): Background threshold. Defaults to 10.
        angular_percentile (float, optional): Percentile for angle selection. Defaults to 99.
        concentration_clip (tuple, optional): Min and max values to clip concentrations.
            Defaults to (0.0, np.inf) - no clipping of maximum.

    Returns:
        np.ndarray: Normalized RGB image with same shape and dtype as input.

    Example:
        >>> # Normalize to own stain vectors
        >>> patch = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> normalized = macenko_stain_normalize(patch)
        >>> print(normalized.shape, normalized.dtype)  # (256, 256, 3) uint8

        >>> # Normalize to reference stain vectors
        >>> reference = np.array([[0.5, 0.6, 0.7], [0.8, 0.2, 0.1]])
        >>> normalized = macenko_stain_normalize(patch, reference_stain_vectors=reference)
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)

    h, w, c = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)

    # Convert to optical density
    od = -np.log((pixels + 1) / 256.0)

    # Get stain vectors from image
    source_stain_vectors = np.array(
        get_macenko_vectors(rgb_image, luminosity_threshold, angular_percentile)
    )

    # Use reference vectors if provided, otherwise use source vectors
    if reference_stain_vectors is None:
        reference_stain_vectors = source_stain_vectors

    # Calculate concentrations for both source and reference
    source_conc = get_macenko_concentrations(
        rgb_image, source_stain_vectors, luminosity_threshold
    )
    
    # Get maximum concentrations from source for normalization
    source_max_conc = np.percentile(source_conc, 99, axis=0)

    # Reshape for processing
    source_conc_reshaped = source_conc.reshape(h * w, 2)

    # Clip concentrations
    source_conc_clipped = np.clip(
        source_conc_reshaped,
        concentration_clip[0],
        concentration_clip[1]
    )

    # Reconstruct image using reference stain vectors
    # OD_normalized = C_clipped * reference_stain_vectors.T
    od_normalized = np.dot(
        source_conc_clipped,
        reference_stain_vectors
    )

    # Convert back from optical density to RGB
    # RGB = 256 * exp(-OD)
    rgb_normalized = 256.0 * np.exp(-od_normalized)
    rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    # Reshape back to image dimensions
    rgb_normalized = rgb_normalized.reshape(h, w, 3)

    return rgb_normalized


def estimate_reference_stain_vectors(
    image_list: list,
    luminosity_threshold: int = 10,
    angular_percentile: float = 99
) -> np.ndarray:
    """
    Estimate reference stain vectors from a collection of images.

    Useful for creating a batch-level or dataset-level reference for consistent
    stain normalization across multiple slides.

    Args:
        image_list (list): List of RGB images (numpy arrays) with shape (H, W, 3).
        luminosity_threshold (int, optional): Background threshold. Defaults to 10.
        angular_percentile (float, optional): Percentile for angle selection. Defaults to 99.

    Returns:
        np.ndarray: Average stain vectors with shape (2, 3).

    Example:
        >>> patches = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
        >>> ref_vectors = estimate_reference_stain_vectors(patches)
        >>> print(ref_vectors.shape)  # (2, 3)
    """
    stain_vectors_list = []

    for img in image_list:
        try:
            stain_vec_1, stain_vec_2 = get_macenko_vectors(
                img, luminosity_threshold, angular_percentile
            )
            stain_vectors_list.append(stain_vec_1)
            stain_vectors_list.append(stain_vec_2)
        except ValueError:
            # Skip images with no foreground
            continue

    if len(stain_vectors_list) == 0:
        raise ValueError("No valid stain vectors could be extracted from image list")

    # Average the stain vectors
    avg_stain_vectors = np.mean(stain_vectors_list, axis=0)

    # Normalize
    avg_stain_vectors = avg_stain_vectors / np.linalg.norm(avg_stain_vectors)

    # Return as (2, 3) array
    return np.array([avg_stain_vectors, np.array([1.0, 1.0, 1.0]) - avg_stain_vectors])
