from pathlib import Path
from typing import Tuple, Optional
import os
import importlib

# Optional cucim backend
try:
    _mod_cucim = importlib.import_module('cucim')
    CuImage = getattr(_mod_cucim, 'CuImage', None)
    _HAS_CUCIM = CuImage is not None
except Exception:
    CuImage = None
    _HAS_CUCIM = False

# OpenSlide backend (commonly installed via openslide-python)
try:
    from openslide import OpenSlide
    _HAS_OPENSLIDE = True and (OpenSlide is not None)
except Exception:
    OpenSlide = None
    _HAS_OPENSLIDE = False

# Optional TiffSlide backend
try:
    _mod_tiffslide = importlib.import_module('tiffslide')
    TiffSlide = getattr(_mod_tiffslide, 'TiffSlide', None)
    _HAS_TIFFSLIDE = TiffSlide is not None
except Exception:
    TiffSlide = None
    _HAS_TIFFSLIDE = False


class WSIReader:
    """Small wrapper that exposes a minimal OpenSlide-like API.

    Notes:
    - This wrapper focuses on level=0 use (high-resolution). If callers
      require multiple pyramid levels, they should use OpenSlide directly.
    - For CuImage backend, read_region returns a numpy array which is
      converted to a PIL.Image to match OpenSlide behavior.
    - Supports context manager protocol for proper resource cleanup.
    """
    def __init__(self, path: Path, backend: str, reader):
        self.path = Path(path)
        self.backend = backend
        self._r = reader
        self._closed = False

        if backend == 'cucim':
            # CuImage exposes dimensions differently across versions.
            # Try multiple patterns in a safe order.
            w = h = None
            # 1) Methods
            if hasattr(self._r, 'get_width') and hasattr(self._r, 'get_height'):
                try:
                    w = int(self._r.get_width())
                    h = int(self._r.get_height())
                except Exception:
                    w = h = None
            # 2) Attributes
            if (w is None or h is None) and hasattr(self._r, 'width') and hasattr(self._r, 'height'):
                try:
                    w = int(getattr(self._r, 'width'))
                    h = int(getattr(self._r, 'height'))
                except Exception:
                    w = h = None
            # 3) Shape
            if (w is None or h is None) and hasattr(self._r, 'shape'):
                try:
                    sh = getattr(self._r, 'shape')
                    if len(sh) >= 2:
                        h = int(sh[0])
                        w = int(sh[1])
                except Exception:
                    w = h = None
            # 4) get_dimensions / size
            if (w is None or h is None) and hasattr(self._r, 'get_dimensions'):
                try:
                    dims = self._r.get_dimensions()
                    if isinstance(dims, (tuple, list)) and len(dims) >= 2:
                        w = int(dims[0])
                        h = int(dims[1])
                except Exception:
                    w = h = None
            if (w is None or h is None) and hasattr(self._r, 'size'):
                try:
                    s = self._r.size
                    if isinstance(s, (tuple, list)) and len(s) >= 2:
                        w = int(s[0])
                        h = int(s[1])
                except Exception:
                    w = h = None
            if w is None or h is None:
                raise RuntimeError('Unable to determine CuImage dimensions')
            self.dimensions = (w, h)
            # only level 0 supported in this lightweight wrapper
            self.level_dimensions = [self.dimensions]
            self.level_downsamples = [1.0]
        else:
            # OpenSlide/TiffSlide-like object
            self.dimensions = self._r.dimensions
            # level_dimensions and downsamples should exist
            try:
                self.level_dimensions = list(self._r.level_dimensions)
            except Exception:
                self.level_dimensions = [self.dimensions]
            try:
                self.level_downsamples = list(self._r.level_downsamples)
            except Exception:
                self.level_downsamples = [1.0 for _ in self.level_dimensions]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close resources."""
        self.close()
        return False

    def close(self):
        """Close the underlying WSI reader and free resources."""
        if self._closed:
            return
        try:
            if hasattr(self._r, 'close'):
                self._r.close()
        except Exception:
            pass
        self._closed = True
        self._r = None

    def __del__(self):
        """Destructor - ensure resources are freed."""
        self.close()

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int], device='cpu'):
        """Read a region and return a PIL.Image (like OpenSlide.read_region).
        
        Args:
            location: (x, y) tuple for top-left corner
            level: pyramid level to read from
            size: (width, height) tuple
            device: 'cpu' or 'cuda' - where to store the image data (CuCIM only)
                   When device='cuda', returns GPU array if possible for downstream GPU processing
        """
        if self._closed:
            raise RuntimeError("Cannot read from closed WSIReader")
        x, y = location
        w, h = size
        if self.backend == 'cucim':
            # CuImage.read_region may return different types depending on version/config.
            # Allow caller to specify device for GPU acceleration
            arr = None
            try:
                arr = self._r.read_region(location=(x, y), size=(w, h), level=level, device=device)
            except TypeError:
                # Older CuCIM versions may not support device parameter
                try:
                    arr = self._r.read_region(location=(x, y), size=(w, h), level=level)
                except Exception:
                    arr = None
            except Exception:
                try:
                    arr = self._r.read_region(location=(x, y), size=(w, h), level=level, device='cpu')
                except Exception:
                    arr = None

            def _to_numpy(a):
                import numpy as _np
                # If already numpy array, return it
                if isinstance(a, _np.ndarray):
                    return a
                # Check if it's a CuPy array (GPU memory)
                try:
                    import cupy as cp  # type: ignore
                except ImportError:
                    cp = None  # type: ignore
                try:
                    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore
                        # Transfer from GPU to CPU
                        return cp.asnumpy(a)  # type: ignore
                except Exception:
                    pass
                # Try numpy.asarray first
                try:
                    na = _np.asarray(a)
                    if isinstance(na, _np.ndarray) and na.size > 0:
                        return na
                except Exception:
                    pass
                # Try common CuCIM/array methods
                for meth in ('copy_to_host', 'to_host', 'to_array', 'to_numpy', 'get', '__array__'):
                    try:
                        fn = getattr(a, meth, None)
                        if fn is None:
                            continue
                        na = fn() if meth != '__array__' else _np.array(a)
                        if isinstance(na, _np.ndarray) and na.size > 0:
                            return na
                    except Exception:
                        continue
                return None

            try:
                import numpy as _np
                from PIL import Image
                
                # If device='cuda' and arr is already a GPU array, keep it on GPU longer
                # Only convert to CPU when we must create a PIL.Image for saving
                # This allows downstream GPU processing (e.g., normalization) to work on GPU data
                try:
                    import cupy as cp  # type: ignore
                except ImportError:
                    cp = None  # type: ignore
                try:
                    is_gpu_array = (cp is not None) and isinstance(arr, cp.ndarray)  # type: ignore
                except Exception:
                    is_gpu_array = False
                
                # For GPU arrays requested via device='cuda', we still need to convert to PIL
                # for compatibility with the current pipeline (saving PNGs)
                # A future optimization could keep data on GPU until the last moment
                arr_np = _to_numpy(arr)
                if arr_np is None:
                    # As last resort, request CPU explicitly again
                    try:
                        arr = self._r.read_region(location=(x, y), size=(w, h), level=level, device='cpu')
                        arr_np = _to_numpy(arr)
                    except Exception:
                        arr_np = None
                if arr_np is None:
                    # Give up and return whatever came back
                    return arr
                # Normalize dtype for PIL
                if arr_np.dtype.kind in ('f',):
                    # Float images: assume 0..1 or 0..255 range
                    m = _np.nanmax(arr_np) if arr_np.size > 0 else 1.0
                    if m <= 1.0 + 1e-6:
                        arr_np = (arr_np * 255.0).clip(0, 255)
                    arr_np = arr_np.astype(_np.uint8)
                elif arr_np.dtype == _np.uint16:
                    # Scale 16-bit to 8-bit
                    arr_np = (arr_np / 257).astype(_np.uint8)
                elif arr_np.dtype.kind not in ('u', 'i'):
                    # Any other types -> uint8 best effort
                    arr_np = arr_np.astype(_np.uint8, copy=False)
                # Ensure (H,W,3) for PIL
                if arr_np.ndim == 3 and arr_np.shape[2] >= 3:
                    return Image.fromarray(arr_np[:, :, :3])
                elif arr_np.ndim == 2:
                    return Image.fromarray(arr_np).convert('RGB')
                else:
                    # If different channel order, try to squeeze/expand
                    if arr_np.ndim == 3 and arr_np.shape[0] in (3, 4) and arr_np.shape[2] == 1:
                        # Likely (C,H,W) with C in first dim
                        arr_np = _np.transpose(arr_np, (1, 2, 0))
                        return Image.fromarray(arr_np[:, :, :3])
                    return Image.fromarray(arr_np)
            except Exception:
                return arr
        else:
            # OpenSlide already returns a PIL.Image
            return self._r.read_region((x, y), level, (w, h))


def _env_backend_preference() -> Optional[str]:
    """Read backend preference from environment variable HER2_WSI_BACKEND.

    Returns 'openslide', 'cucim', 'tiffslide', or None.
    """
    val = os.environ.get('HER2_WSI_BACKEND') or os.environ.get('WSI_BACKEND')
    if not val:
        return None
    val = str(val).strip().lower()
    if val in ('openslide', 'cucim', 'tiffslide'):
        return val
    return None


def load_wsi(wsi_path, prefer_backend: Optional[str] = None, force_backend: Optional[str] = None):
    """Load a WSI and return a `WSIReader` wrapper.

    Selection order:
    - If force_backend is provided ('openslide', 'cucim', 'tiffslide'), try only that backend.
    - Else, if HER2_WSI_BACKEND env var is set, try that backend first then fallback.
    - Else, if prefer_backend is provided, try it first then fallback.
    - Else, prefer CuCIM when available, otherwise OpenSlide, then TiffSlide.
    """
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    # Normalize backend strings
    def norm(b):
        return b if b in (None, 'openslide', 'cucim', 'tiffslide') else None

    prefer_backend = norm(prefer_backend)
    force_backend = norm(force_backend)
    env_backend = _env_backend_preference()

    order = []
    if force_backend:
        order = [force_backend]
    else:
        if env_backend:
            order.append(env_backend)
        if prefer_backend:
            order.append(prefer_backend)
        # Default priority
        order.extend(['cucim', 'openslide', 'tiffslide'])

    # Remove duplicates while preserving order
    seen = set()
    ordered_backends = []
    for b in order:
        if b and b not in seen:
            seen.add(b)
            ordered_backends.append(b)

    last_error = None
    for backend in ordered_backends:
        if backend == 'cucim' and _HAS_CUCIM and (CuImage is not None):
            try:
                reader = CuImage(str(wsi_path))
                return WSIReader(wsi_path, 'cucim', reader)
            except Exception as e:
                last_error = e
                print(f"CuCIM failed to open {wsi_path.name}: {e}")
        elif backend == 'openslide' and _HAS_OPENSLIDE and (OpenSlide is not None):
            try:
                reader = OpenSlide(str(wsi_path))
                return WSIReader(wsi_path, 'openslide', reader)
            except Exception as e:
                last_error = e
                print(f"OpenSlide failed to open {wsi_path.name}: {e}")
        elif backend == 'tiffslide' and _HAS_TIFFSLIDE and (TiffSlide is not None):
            try:
                reader = TiffSlide(str(wsi_path))
                return WSIReader(wsi_path, 'tiffslide', reader)
            except Exception as e:
                last_error = e
                print(f"TiffSlide failed to open {wsi_path.name}: {e}")

    if last_error:
        print(f"Failed to load WSI with available backends ({ordered_backends}): {wsi_path}\nLast error: {last_error}")
    else:
        print(f"Failed to load WSI with available backends: {wsi_path}")
    return None
