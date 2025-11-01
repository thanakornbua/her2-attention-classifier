import numpy as np


def macenko_normalization(image, target_concentrations=None, target_stains=None):
    """
    Macenko stain normalization for H&E stained histopathology images.
    
    Normalizes staining variations across different slides to a reference appearance.
    This is critical for consistent HER2 scoring in clinical workflows.
    
    Args:
        image (np.ndarray): RGB image (H x W x 3), uint8 or float [0-255]
        target_concentrations (np.ndarray, optional): Reference stain concentrations (2 x N)
        target_stains (np.ndarray, optional): Reference stain matrix (3 x 2)
    
    Returns:
        np.ndarray: Normalized RGB image (same shape and dtype as input)
    
    References:
        Macenko et al. "A method for normalizing histology slides for 
        quantitative analysis" (ISBI 2009)
    
    Practical notes:
        - Handles white/background pixels robustly
        - Uses percentile-based thresholding (clinical standard)
        - Returns original dtype for pipeline compatibility
    """
    # Store original dtype for return
    original_dtype = image.dtype
    
    # Handle edge cases
    if image.size == 0:
        return image
    
    # Work with float32 internally
    image_float = image.astype(np.float32) if image.dtype != np.float32 else image
    
    # Convert RGB to optical density (OD)
    image_od = rgb_to_od(image_float)
    
    # Extract stain matrix and concentrations from source image
    stain_matrix_source, concentrations_source = get_stain_matrix(image_od)
    
    # Use default reference if not provided
    if target_stains is None or target_concentrations is None:
        target_stains, target_concentrations = get_default_reference()
    
    # Reconstruct image with target stains using SOURCE concentrations
    # Keep the tissue structure (concentrations) but change color (stains)
    normalized_od = (target_stains @ concentrations_source).T.reshape(image_od.shape)
    normalized_image = od_to_rgb(normalized_od)
    
    # Free intermediate arrays
    del image_od, stain_matrix_source, concentrations_source, normalized_od
    
    # Ensure output matches input dtype and range
    normalized_image = np.clip(normalized_image, 0, 255)
    result = normalized_image.astype(original_dtype)
    
    # Free normalized_image before return
    del normalized_image
    
    return result


def rgb_to_od(image):
    """
    Convert RGB image to Optical Density (OD) space.
    
    OD = -log10(I/I0), where I is intensity and I0 is incident light (255).
    
    Args:
        image (np.ndarray): RGB image, float32 [0-255]
    
    Returns:
        np.ndarray: Optical density values (H x W x 3)
    """
    # Convert to optical density: OD = -log10(I/255), avoiding log(0)
    return -np.log10((image + 1.0) / 256.0)


def od_to_rgb(od):
    """
    Convert Optical Density (OD) back to RGB space.
    
    I = 255 * 10^(-OD)
    
    Args:
        od (np.ndarray): Optical density values
    
    Returns:
        np.ndarray: RGB image, float32 [0-255]
    """
    return 255.0 * np.power(10.0, -od)


def get_stain_matrix(image_od, alpha=1, beta=0.15):
    """
    Extract H&E stain matrix using Macenko's method.
    
    Uses Singular Value Decomposition (SVD) to find the two dominant stain vectors
    (Hematoxylin and Eosin) in the optical density space.
    
    Args:
        image_od (np.ndarray): Optical density image (H x W x 3)
        alpha (float): Percentile for robust angle selection (default: 1%)
        beta (float): OD threshold to exclude background (default: 0.15)
    
    Returns:
        stain_matrix (np.ndarray): 3x2 matrix of stain vectors
        concentrations (np.ndarray): 2xN matrix of stain concentrations
    
    Practical notes:
        - Beta=0.15 is clinically validated for H&E slides
        - Alpha percentile handles outliers robustly
    """
    # Reshape to (N_pixels, 3)
    od_flat = image_od.reshape(-1, 3)
    
    # Remove background pixels (low OD = white/unstained)
    tissue_mask = (od_flat > beta).all(axis=1)
    od_tissue = od_flat[tissue_mask]
    
    # Handle edge case: no tissue pixels
    if od_tissue.shape[0] < 2:
        # Return identity-like stain matrix
        stain_matrix = np.array([[0.644, 0.716], 
                                  [0.092, 0.954], 
                                  [0.759, 0.650]])
        concentrations = np.zeros((2, od_flat.shape[0]))
        return stain_matrix, concentrations
    
    # Singular Value Decomposition to find principal stain directions
    # od_tissue.T is (3, N), so U is (3, 3) - the left singular vectors
    U, _, _ = np.linalg.svd(od_tissue.T, full_matrices=False)
    U = U[:, :2]  # (3, 2) - the two principal directions in 3D OD space
    
    # Project tissue OD onto this 2D plane
    projected = od_tissue @ U  # (N, 3) @ (3, 2) = (N, 2)
    
    # Find extreme angles (robust to outliers using percentiles)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)
    
    # Compute stain vectors from extreme angles in the 2D projected space
    # Then map back to 3D OD space using U basis
    v1 = np.cos(min_angle) * U[:, 0] + np.sin(min_angle) * U[:, 1]
    v2 = np.cos(max_angle) * U[:, 0] + np.sin(max_angle) * U[:, 1]
    
    # Ensure H (hematoxylin) has higher OD in blue channel
    if v1[2] > v2[2]:
        stain_matrix = np.column_stack([v1, v2])
    else:
        stain_matrix = np.column_stack([v2, v1])
    
    # Normalize stain vectors
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)
    
    # Calculate concentrations: C = inv(S) @ OD
    concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
    concentrations = np.maximum(concentrations, 0)  # Non-negative constraint
    
    return stain_matrix, concentrations


def get_default_reference():
    """
    Return default reference H&E stain matrix and concentrations.
    
    These values are derived from standard H&E reference images used in
    clinical digital pathology pipelines.
    
    Returns:
        stain_matrix (np.ndarray): 3x2 reference stain vectors
        concentrations (np.ndarray): 2x1 reference concentrations
    
    Practical notes:
        - Based on widely-used clinical standards
        - H (hematoxylin): blue/purple, E (eosin): pink/red
    """
    # Standard H&E stain vectors (column-wise: [H, E])
    stain_matrix = np.array([
        [0.644, 0.716],  # Red channel
        [0.092, 0.954],  # Green channel
        [0.759, 0.650]   # Blue channel
    ])
    
    # Default reference concentrations (moderate staining)
    concentrations = np.array([[1.0], [1.0]])
    
    return stain_matrix, concentrations