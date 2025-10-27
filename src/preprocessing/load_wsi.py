from pathlib import Path
from typing import Tuple

try:
    from cucim import CuImage
    _HAS_CUCIM = True
except Exception:
    CuImage = None
    _HAS_CUCIM = False

try:
    from openslide import OpenSlide
    _HAS_OPENSLIDE = True
except Exception:
    OpenSlide = None
    _HAS_OPENSLIDE = False


class WSIReader:
    """Small wrapper that exposes a minimal OpenSlide-like API.

    Notes:
    - This wrapper focuses on level=0 use (high-resolution). If callers
      require multiple pyramid levels, they should use OpenSlide directly.
    - For CuImage backend, read_region returns a numpy array which is
      converted to a PIL.Image to match OpenSlide behavior.
    """
    def __init__(self, path: Path, backend: str, reader):
        self.path = Path(path)
        self.backend = backend
        self._r = reader

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
            # OpenSlide-like object
            self.dimensions = self._r.dimensions
            # level_dimensions and downsamples should exist on OpenSlide
            try:
                self.level_dimensions = list(self._r.level_dimensions)
            except Exception:
                self.level_dimensions = [self.dimensions]
            try:
                self.level_downsamples = list(self._r.level_downsamples)
            except Exception:
                self.level_downsamples = [1.0 for _ in self.level_dimensions]

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]):
        """Read a region and return a PIL.Image (like OpenSlide.read_region)."""
        x, y = location
        w, h = size
        if self.backend == 'cucim':
            # CuImage.read_region may return different types depending on version/config.
            # Request CPU output; then coerce to numpy and PIL.Image.
            arr = None
            try:
                arr = self._r.read_region(location=(x, y), size=(w, h), level=level)
                if arr is None:
                    arr = self._r.read_region(location=(x, y), size=(w, h), level=level, device='cpu')
            except Exception:
                try:
                    arr = self._r.read_region(location=(x, y), size=(w, h), level=level, device='cpu')
                except Exception:
                    arr = None

            def _to_numpy(a):
                import numpy as _np
                # Try numpy.asarray first
                try:
                    na = _np.asarray(a)
                    if isinstance(na, _np.ndarray) and na.size > 0:
                        return na
                except Exception:
                    pass
                # Try common CuCIM/array methods
                for meth in ('copy_to_host', 'to_host', 'to_array', 'to_numpy', '__array__'):
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
                # Ensure (H,W,3) for PIL
                if arr_np.ndim == 3 and arr_np.shape[2] >= 3:
                    return Image.fromarray(arr_np[:, :, :3])
                elif arr_np.ndim == 2:
                    return Image.fromarray(arr_np).convert('RGB')
                else:
                    return Image.fromarray(arr_np)
            except Exception:
                return arr
        else:
            # OpenSlide already returns a PIL.Image
            return self._r.read_region((x, y), level, (w, h))


def load_wsi(wsi_path):
    """Load a WSI and return a `WSIReader` wrapper using CuCIM if available.

    Raises FileNotFoundError if the file does not exist. Returns None on failure.
    """
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    # Prefer CuCIM for speed if present
    if _HAS_CUCIM:
        try:
            reader = CuImage(str(wsi_path))
            return WSIReader(wsi_path, 'cucim', reader)
        except Exception as e:
            # fallback to OpenSlide
            print(f"CuCIM available but failed to open {wsi_path.name}: {e}")

    if _HAS_OPENSLIDE:
        try:
            reader = OpenSlide(str(wsi_path))
            return WSIReader(wsi_path, 'openslide', reader)
        except Exception as e:
            print(f"OpenSlide failed to open {wsi_path.name}: {e}")

    # If neither backend worked, return None
    print(f"Failed to load WSI with available backends: {wsi_path}")
    return None