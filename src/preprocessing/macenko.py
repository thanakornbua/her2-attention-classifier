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

    This function estimates the two primary stain vectors (Hematoxylin and Eosin) from a
    histology image by:
    1. Converting RGB to optical density (OD) space
    2. Performing SVD to identify the principal plane
    3. Projecting OD values onto this plane
    4. Converting to polar coordinates
    5. Selecting extreme angles (e.g., 1st and 99th percentiles) as stain vectors

    Args:
        rgb_image (np.ndarray): RGB image as numpy array with shape (height, width, 3).
            Values should be in range [0, 255] with dtype uint8.
        luminosity_threshold (int, optional): Pixels with luminosity below this value
            are considered background and excluded. Defaults to 10.
        angular_percentile (float, optional): Percentile for selecting extreme angles
            in the angular distribution. Defaults to 99 (99th percentile).

    Returns:
        tuple: (stain_vector_H, stain_vector_E) - Two normalized stain vectors
            Each vector has shape (3,) and represents one stain direction in OD space.
            Order: H (Hematoxylin) is typically first, E (Eosin) is second.

    Raises:
        ValueError: If input image shape is invalid.
        ValueError: If no foreground pixels found (image too bright/white).

    Example:
        >>> patch = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> stain_H, stain_E = get_macenko_vectors(patch)
        >>> print(stain_H.shape, stain_E.shape)  # (3,) (3,)
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

    # MACENKO STEP 1: SVD to find principal plane
    # SVD of (N, 3) matrix returns U(N,3), s(3,), Vt(3,3)
    # V (columns of Vt.T) are the orthonormal basis vectors
    _, _, Vt = np.linalg.svd(od_normalized, full_matrices=False)
    V = Vt.T  # Shape (3, 3)

    # MACENKO STEP 2: Project OD values onto the principal plane (first 2 components)
    # This projects all normalized OD vectors onto the plane spanned by V[:, 0] and V[:, 1]
    projections = np.dot(od_normalized, V[:, :2])  # Shape (N, 2)

    # MACENKO STEP 3: Convert to polar coordinates in the 2D plane
    # Calculate angles for each projected point
    angles = np.arctan2(projections[:, 1], projections[:, 0])  # Shape (N,)

    # MACENKO STEP 4: Select extreme angles
    # Find the angles at the lower and upper percentiles
    # These extreme angles correspond to the two stain directions
    lower_percentile = 100 - angular_percentile  # e.g., 1 for 99th percentile
    upper_percentile = angular_percentile          # e.g., 99 for 99th percentile

    angle_low = np.percentile(angles, lower_percentile)
    angle_high = np.percentile(angles, upper_percentile)

    # Convert angles back to stain vectors in the original 3D OD space
    # Create unit vectors at these extreme angles in the 2D plane
    stain_plane_1 = np.array([np.cos(angle_low), np.sin(angle_low)])
    stain_plane_2 = np.array([np.cos(angle_high), np.sin(angle_high)])

    # Project back to 3D OD space using the basis vectors V[:, :2]
    stain_vec_H = np.dot(V[:, :2], stain_plane_1)
    stain_vec_E = np.dot(V[:, :2], stain_plane_2)

    # Normalize to unit vectors
    stain_vec_H = stain_vec_H / np.linalg.norm(stain_vec_H)
    stain_vec_E = stain_vec_E / np.linalg.norm(stain_vec_E)

    # Ensure consistent orientation (positive first component)
    if stain_vec_H[0] < 0:
        stain_vec_H = -stain_vec_H
    if stain_vec_E[0] < 0:
        stain_vec_E = -stain_vec_E

    return stain_vec_H, stain_vec_E


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

    This function creates a dataset-level reference by:
    1. Extracting stain vectors (Hematoxylin and Eosin) from each image
    2. Maintaining separate lists for H and E vectors
    3. Computing the mean H vector and mean E vector independently
    4. Returning the average stain matrix for consistent normalization

    Useful for creating a robust, batch-level or dataset-level reference for
    consistent stain normalization across multiple slides and batches.

    Args:
        image_list (list): List of RGB images (numpy arrays) with shape (H, W, 3).
        luminosity_threshold (int, optional): Background threshold. Defaults to 10.
        angular_percentile (float, optional): Percentile for angle selection. Defaults to 99.

    Returns:
        np.ndarray: Reference stain matrix with shape (2, 3).
                   Row 0: Average Hematoxylin vector
                   Row 1: Average Eosin vector

    Raises:
        ValueError: If no valid stain vectors can be extracted from the image list.

    Example:
        >>> patches = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
        >>> ref_vectors = estimate_reference_stain_vectors(patches)
        >>> print(ref_vectors.shape)  # (2, 3)
        >>> print(ref_vectors[0])  # Average Hematoxylin vector
        >>> print(ref_vectors[1])  # Average Eosin vector
    """
    stain_vectors_H = []  # List of Hematoxylin vectors
    stain_vectors_E = []  # List of Eosin vectors

    for img in image_list:
        try:
            stain_vec_H, stain_vec_E = get_macenko_vectors(
                img, luminosity_threshold, angular_percentile
            )
            # Keep H and E separated in different lists
            stain_vectors_H.append(stain_vec_H)
            stain_vectors_E.append(stain_vec_E)
        except ValueError:
            # Skip images with no foreground or insufficient stain separation
            continue

    if len(stain_vectors_H) == 0 or len(stain_vectors_E) == 0:
        raise ValueError("No valid stain vectors could be extracted from image list")

    # Average H and E vectors independently
    avg_stain_vec_H = np.mean(stain_vectors_H, axis=0)
    avg_stain_vec_E = np.mean(stain_vectors_E, axis=0)

    # Normalize to unit vectors
    avg_stain_vec_H = avg_stain_vec_H / np.linalg.norm(avg_stain_vec_H)
    avg_stain_vec_E = avg_stain_vec_E / np.linalg.norm(avg_stain_vec_E)

    # Return as (2, 3) reference stain matrix
    return np.array([avg_stain_vec_H, avg_stain_vec_E])

