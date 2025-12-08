"""
WSI Reader Module

Supports reading whole slide images using OpenSlide or CUCIM (GPU-accelerated when available).
Provides a unified interface for extracting patches from whole slide images with automatic
backend detection and fallback mechanisms.

Authors: T. Buathongtanakarn (2025) et al.
License: MIT
"""

import numpy as np
from typing import Tuple, Union
import warnings
import sys


def check_backends() -> dict:
    """
    Check availability of WSI reading backends.

    Returns:
        dict: Status of each backend with diagnostic information.

    Example:
        >>> status = check_backends()
        >>> print(status)
    """
    status = {
        'openslide': {'available': False, 'error': None},
        'cucim': {'available': False, 'error': None}
    }

    # Check OpenSlide
    try:
        import openslide
        status['openslide']['available'] = True
        status['openslide']['version'] = openslide.__version__ if hasattr(openslide, '__version__') else 'Unknown'
    except ImportError as e:
        status['openslide']['error'] = f"Not installed: {e}"
    except Exception as e:
        status['openslide']['error'] = f"Import error: {e}"

    # Check CUCIM
    try:
        import cucim
        status['cucim']['available'] = True
        status['cucim']['version'] = cucim.__version__ if hasattr(cucim, '__version__') else 'Unknown'
    except ImportError as e:
        status['cucim']['error'] = f"Not installed: {e}"
    except Exception as e:
        status['cucim']['error'] = f"Import error: {e}"

    return status


class WSIReader:
    """
    Whole Slide Image Reader with automatic backend selection.

    Provides a unified interface for reading regions from whole slide images (WSI),
    supporting both OpenSlide and CUCIM (GPU-accelerated) backends. Automatically
    selects the best available backend with configurable preference.

    Attributes:
        slide_path (str): Path to the whole slide image file.
        backend (str): The backend being used ('openslide' or 'cucim').
        slide: The slide object from the respective backend.
    """
    
    def __init__(self, slide_path: str, prefer_cucim: bool = True):
        """
        Initialize WSI Reader.

        Attempts to load the slide with the preferred backend, falling back to
        alternative backends if necessary.

        Args:
            slide_path (str): Path to the whole slide image file.
            prefer_cucim (bool, optional): If True, attempts CUCIM first,
                then falls back to OpenSlide. Defaults to True.

        Raises:
            ImportError: If neither OpenSlide nor CUCIM are available.
            FileNotFoundError: If the slide file doesn't exist.
        """
        from pathlib import Path
        
        self.slide_path = slide_path
        self.slide = None
        self.backend = None
        
        # Verify file exists
        slide_file = Path(slide_path)
        if not slide_file.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
        
        # Try to load the slide with preferred backend
        if prefer_cucim:
            if self._try_cucim():
                return
            if self._try_openslide():
                return
        else:
            if self._try_openslide():
                return
            if self._try_cucim():
                return
        
        # If we reach here, no backend worked
        status = check_backends()
        error_msg = "Failed to load slide with any available backend.\n\n"
        error_msg += f"File: {slide_path}\n"
        error_msg += f"Exists: {slide_file.exists()}\n"
        error_msg += f"Size: {slide_file.stat().st_size if slide_file.exists() else 'N/A'} bytes\n\n"
        error_msg += "Backend Status:\n"
        
        for backend_name, backend_status in status.items():
            if backend_status['available']:
                error_msg += f"  - {backend_name}: Available but failed to load slide\n"
            else:
                error_msg += f"  - {backend_name}: {backend_status['error']}\n"
        
        error_msg += "\nInstallation instructions:\n"
        error_msg += "  pip install openslide-python\n"
        error_msg += "  # or for GPU acceleration:\n"
        error_msg += "  pip install cucim\n"
        
        raise ImportError(error_msg)
    
    def _try_cucim(self) -> bool:
        """Try to initialize with CUCIM backend"""
        try:
            from cucim import CuImage
            self.slide = CuImage(self.slide_path)
            self.backend = 'cucim'
            return True
        except ImportError:
            return False
        except Exception as e:
            warnings.warn(f"CUCIM failed to load slide: {e}")
            return False
    
    def _try_openslide(self) -> bool:
        """Try to initialize with OpenSlide backend."""
        try:
            import openslide
            self.slide = openslide.OpenSlide(self.slide_path)
            self.backend = 'openslide'
            return True
        except ImportError as e:
            # OpenSlide not installed
            return False
        except Exception as e:
            # OpenSlide installed but failed to load the slide
            warnings.warn(f"OpenSlide available but failed to load slide: {type(e).__name__}: {e}")
            return False
    
    def read_region(
        self, 
        x: int, 
        y: int, 
        level: int, 
        size: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Read a region from the whole slide image.

        Extracts a rectangular patch from the slide at the specified coordinates,
        pyramid level, and size. Coordinates are always in the level 0 reference frame.

        Args:
            x (int): X coordinate in level 0 reference frame.
            y (int): Y coordinate in level 0 reference frame.
            level (int): Pyramid level to read from (0 is highest resolution).
            size (int or tuple of int): Size of the region to read. If int, reads
                a square region (size, size). If tuple, reads rectangular region
                (width, height).

        Returns:
            np.ndarray: RGB image as numpy array with shape (height, width, 3).
                Values are in range [0, 255] with dtype uint8.
        """
        # Handle size parameter
        if isinstance(size, int):
            width, height = size, size
        else:
            width, height = size
        
        if self.backend == 'cucim':
            return self._read_region_cucim(x, y, level, width, height)
        elif self.backend == 'openslide':
            return self._read_region_openslide(x, y, level, width, height)
        else:
            raise RuntimeError("No valid backend available")
    
    def _read_region_cucim(
        self, 
        x: int, 
        y: int, 
        level: int, 
        width: int, 
        height: int
    ) -> np.ndarray:
        """Read region using CUCIM backend"""
        # CUCIM uses (location, size, level) format
        region = self.slide.read_region(
            location=(x, y),
            size=(width, height),
            level=level
        )
        # Convert to numpy array and ensure RGB format
        img_array = np.array(region)
        
        # CUCIM typically returns RGB directly
        if img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Drop alpha channel
        
        return img_array
    
    def _read_region_openslide(
        self, 
        x: int, 
        y: int, 
        level: int, 
        width: int, 
        height: int
    ) -> np.ndarray:
        """Read region using OpenSlide backend"""
        # OpenSlide read_region returns PIL Image in RGBA format
        pil_img = self.slide.read_region(
            location=(x, y),
            level=level,
            size=(width, height)
        )
        
        # Convert PIL Image to numpy array and RGB
        img_array = np.array(pil_img)
        
        # OpenSlide returns RGBA, convert to RGB
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        return img_array
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Get dimensions of the slide at level 0 (highest resolution).

        Returns:
            tuple: (width, height) of the slide at level 0.
        """
        if self.backend == 'cucim':
            # CUCIM returns shape as (height, width, channels)
            shape = self.slide.shape
            return (shape[1], shape[0])
        elif self.backend == 'openslide':
            return self.slide.dimensions
        else:
            raise RuntimeError("No valid backend available")
    
    @property
    def level_count(self) -> int:
        """
        Get number of pyramid levels in the slide.

        Returns:
            int: Number of pyramid levels available.
        """
        if self.backend == 'cucim':
            return self.slide.resolutions['level_count']
        elif self.backend == 'openslide':
            return self.slide.level_count
        else:
            raise RuntimeError("No valid backend available")
    
    @property
    def level_dimensions(self) -> list:
        """
        Get dimensions at each pyramid level.

        Returns:
            list: List of (width, height) tuples for each level.
        """
        if self.backend == 'cucim':
            level_dims = []
            for i in range(self.level_count):
                shape = self.slide.resolutions['level_dimensions'][i]
                level_dims.append((shape[0], shape[1]))
            return level_dims
        elif self.backend == 'openslide':
            return self.slide.level_dimensions
        else:
            raise RuntimeError("No valid backend available")
    
    @property
    def level_downsamples(self) -> list:
        """
        Get downsample factor for each pyramid level relative to level 0.

        Returns:
            list: Downsample factors for each level.
        """
        if self.backend == 'cucim':
            return list(self.slide.resolutions['level_downsamples'])
        elif self.backend == 'openslide':
            return self.slide.level_downsamples
        else:
            raise RuntimeError("No valid backend available")
    
    def get_thumbnail(self, size: Union[int, Tuple[int, int]] = 512) -> np.ndarray:
        """
        Get a thumbnail of the entire slide.

        Generates a downsampled view of the complete slide, useful for overview
        and quality assessment.

        Args:
            size (int or tuple, optional): Maximum size for thumbnail (width, height).
                If int, uses square size (size, size). Defaults to 512.

        Returns:
            np.ndarray: RGB thumbnail as numpy array.
        """
        if isinstance(size, int):
            size = (size, size)
        
        if self.backend == 'cucim':
            # Use the lowest resolution level for thumbnail
            lowest_level = self.level_count - 1
            dims = self.level_dimensions[lowest_level]
            thumbnail = self.slide.read_region(
                location=(0, 0),
                size=dims,
                level=lowest_level
            )
            thumb_array = np.array(thumbnail)
            if thumb_array.shape[-1] == 4:
                thumb_array = thumb_array[:, :, :3]
            return thumb_array
        
        elif self.backend == 'openslide':
            thumbnail = self.slide.get_thumbnail(size)
            thumb_array = np.array(thumbnail)
            if thumb_array.shape[-1] == 4:
                thumb_array = thumb_array[:, :, :3]
            return thumb_array
        
        else:
            raise RuntimeError("No valid backend available")
    
    def close(self):
        """
        Close the slide file and release resources.

        Should be called when done reading to ensure proper cleanup.
        Alternatively, use the context manager (with statement) for automatic cleanup.
        """
        if self.slide is not None:
            if self.backend == 'cucim':
                # CUCIM slides are typically closed automatically
                pass
            elif self.backend == 'openslide':
                self.slide.close()
        self.slide = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper resource cleanup."""
        self.close()
    
    def __repr__(self) -> str:
        return (
            f"WSIReader(path='{self.slide_path}', "
            f"backend='{self.backend}', "
            f"dimensions={self.dimensions}, "
            f"levels={self.level_count})"
        )


# Convenience function for quick usage in notebooks
def read_wsi_region(
    slide_path: str,
    x: int,
    y: int,
    level: int = 0,
    size: Union[int, Tuple[int, int]] = 256,
    prefer_cucim: bool = True
) -> np.ndarray:
    """
    Convenience function to read a single region from a WSI file.

    Useful for quick, one-off patch extraction. For multiple reads from the same
    slide, use WSIReader directly to avoid repeated file loading.

    Args:
        slide_path (str): Path to the whole slide image file.
        x (int): X coordinate in level 0 reference frame.
        y (int): Y coordinate in level 0 reference frame.
        level (int, optional): Pyramid level to read from. Defaults to 0.
        size (int or tuple, optional): Size of the region. Defaults to 256.
        prefer_cucim (bool, optional): Prefer CUCIM over OpenSlide if available.
            Defaults to True.

    Returns:
        np.ndarray: RGB image as numpy array with shape (height, width, 3).
    """
    with WSIReader(slide_path, prefer_cucim=prefer_cucim) as reader:
        return reader.read_region(x, y, level, size)
