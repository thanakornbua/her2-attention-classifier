from pathlib import Path
from openslide import OpenSlide

def load_wsi(wsi_path):
    """Load WSI using OpenSlide. Returns slide object or None if failed."""
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")
    try:
        slide = OpenSlide(str(wsi_path))
        _ = slide.dimensions  # Validate
        return slide
    except Exception as e:
        print(f"Error loading '{wsi_path.name}': {e}")
        return None