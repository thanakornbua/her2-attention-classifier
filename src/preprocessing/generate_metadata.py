"""Generate metadata CSV from WSI files and annotations for HER2 classification."""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False


def generate_metadata(wsi_dir, annotation_dir=None, output_csv='metadata.csv', 
                     label_file=None, wsi_extensions=None):
    """
    Generate metadata CSV from WSI files.
    
    Args:
        wsi_dir: Directory with WSI files
        annotation_dir: Directory with annotation files (optional)
        output_csv: Output CSV path
        label_file: CSV with slide_id,label columns
        wsi_extensions: WSI file extensions (default: .svs, .tif, .ndpi, .mrxs)
    
    Returns:
        DataFrame with columns: slide_id, filename, wsi_path, label, width, height, 
                               level_count, annotation_path, file_size_mb, valid
    """
    if wsi_extensions is None:
        wsi_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs']
    
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI directory not found: {wsi_dir}")
    
    # Find WSI files
    wsi_files = []
    for ext in wsi_extensions:
        wsi_files.extend(wsi_dir.rglob(f"*{ext}"))
        wsi_files.extend(wsi_dir.rglob(f"*{ext.upper()}"))
    wsi_files = sorted(set(wsi_files))
    
    if not wsi_files:
        raise ValueError(f"No WSI files found in {wsi_dir}")
    
    # Load labels and annotations
    labels = _load_labels(label_file) if label_file else {}
    annotations = _find_annotations(annotation_dir) if annotation_dir else {}
    
    # Process each WSI
    metadata_list = []
    for wsi_path in tqdm(wsi_files, desc="Processing WSIs"):
        slide_id = wsi_path.stem
        metadata = {
            'slide_id': slide_id,
            'filename': wsi_path.name,
            'wsi_path': str(wsi_path.absolute()),
            'label': labels.get(slide_id),
            'width': None,
            'height': None,
            'level_count': None,
            'annotation_path': annotations.get(slide_id),
            'file_size_mb': round(wsi_path.stat().st_size / (1024**2), 2),
            'valid': False
        }
        
        if OPENSLIDE_AVAILABLE:
            try:
                slide = openslide.open_slide(str(wsi_path))
                metadata['width'], metadata['height'] = slide.dimensions
                metadata['level_count'] = slide.level_count
                metadata['valid'] = True
                slide.close()
            except:
                pass
        else:
            metadata['valid'] = True
        
        metadata_list.append(metadata)
    
    # Save CSV
    df = pd.DataFrame(metadata_list)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Saved {len(df)} slides to {output_csv}")
    print(f"   Valid: {df['valid'].sum()} | With labels: {df['label'].notna().sum()}")
    return df


def _load_labels(label_file):
    """Load labels from CSV or TXT file with flexible column detection."""
    label_file = Path(label_file)
    if not label_file.exists():
        return {}
    
    labels = {}
    
    # Handle CSV files
    if label_file.suffix.lower() == '.csv':
        df = pd.read_csv(label_file)
        # Auto-detect columns (supports variations in naming)
        id_col = next((c for c in df.columns if c.lower() in ['slide_id', 'id', 'filename', 'slide', 'name']), df.columns[0])
        label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'grade', 'score', 'her2']), df.columns[1])
        
        for _, row in df.iterrows():
            slide_id = Path(str(row[id_col])).stem
            labels[slide_id] = row[label_col]
    
    # Handle plain text files (tab/space/comma separated)
    else:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = re.split(r'[,\t\s]+', line, maxsplit=1)
                    if len(parts) == 2:
                        slide_id = Path(parts[0]).stem
                        labels[slide_id] = parts[1]
    
    return labels


def _find_annotations(annotation_dir):
    """Find annotation files and map by slide_id (supports XML, JSON, masks)."""
    annotation_dir = Path(annotation_dir)
    if not annotation_dir.exists():
        return {}
    
    # Support multiple annotation formats
    exts = ['.xml', '.json', '.txt', '.png', '.tif', '.tiff', '.npy']
    files = []
    for ext in exts:
        files.extend(annotation_dir.rglob(f"*{ext}"))
        files.extend(annotation_dir.rglob(f"*{ext.upper()}"))  # Case-insensitive
    
    return {f.stem: str(f.absolute()) for f in files}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate metadata CSV from WSI files")
    parser.add_argument("wsi_dir", help="Directory with WSI files")
    parser.add_argument("--annotation-dir", help="Directory with annotations")
    parser.add_argument("-o", "--output", default="metadata.csv", help="Output CSV")
    parser.add_argument("--label-file", help="CSV with slide_id,label")
    args = parser.parse_args()
    
    generate_metadata(args.wsi_dir, args.annotation_dir, args.output, args.label_file)
