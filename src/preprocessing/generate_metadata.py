"""
Generate metadata CSV from WSI files and annotations for HER2 classification.

This module scans directories containing Whole Slide Images (WSIs) and their 
corresponding annotations/labels, then generates a metadata CSV file for downstream
processing in the MIL pipeline.
"""

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    warnings.warn("OpenSlide not available. WSI dimensions will not be extracted.")


def generate_metadata(wsi_dir, annotation_dir, output_csv, label_file=None, 
                     wsi_extensions=None, verbose=True):
    """
    Generate metadata CSV from available WSIs and labels.
    
    This function performs the following:
    1. Scans wsi_dir for whole slide images
    2. Extracts metadata (dimensions, path, etc.) from each WSI
    3. Matches WSIs with their corresponding labels/annotations
    4. Exports a comprehensive metadata CSV for training
    
    Args:
        wsi_dir (str): Directory containing whole slide images
        annotation_dir (str): Directory containing annotation files (XML, JSON, or text)
                             Can be None if using label_file instead
        output_csv (str): Output path for the generated metadata CSV
        label_file (str, optional): Path to a CSV/text file containing slide_id and labels
                                   Format: slide_id,label (e.g., "slide001,2+")
        wsi_extensions (list, optional): List of WSI file extensions to search for.
                                        Default: ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.vms']
        verbose (bool): Whether to print progress information. Default: True
    
    Returns:
        pd.DataFrame: The generated metadata dataframe
        
    Output CSV Structure:
        - slide_id: Unique identifier extracted from filename
        - filename: Original WSI filename
        - wsi_path: Absolute path to WSI file
        - label: HER2 classification label (0, 1+, 2+, 3+, or numeric)
        - width: WSI width in pixels (level 0)
        - height: WSI height in pixels (level 0)
        - level_count: Number of pyramid levels
        - annotation_path: Path to annotation file (if exists, else None)
        - file_size_mb: File size in megabytes
        - valid: Boolean indicating if WSI can be opened (if OpenSlide available)
    
    Example:
        >>> generate_metadata(
        ...     wsi_dir='data/wsi/',
        ...     annotation_dir='data/annotations/',
        ...     output_csv='data/metadata.csv',
        ...     label_file='data/labels.csv'
        ... )
        
        >>> # With custom extensions
        >>> generate_metadata(
        ...     wsi_dir='data/slides/',
        ...     annotation_dir=None,
        ...     output_csv='metadata.csv',
        ...     label_file='labels.txt',
        ...     wsi_extensions=['.svs', '.ndpi']
        ... )
    """
    
    # Default WSI extensions
    if wsi_extensions is None:
        wsi_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.vms', '.vmu', '.scn']
    
    # Convert to Path objects
    wsi_dir = Path(wsi_dir)
    if annotation_dir:
        annotation_dir = Path(annotation_dir)
    
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI directory not found: {wsi_dir}")
    
    if verbose:
        print(f"Scanning WSI directory: {wsi_dir}")
        print(f"Looking for extensions: {wsi_extensions}")
    
    # Step 1: Find all WSI files
    wsi_files = []
    for ext in wsi_extensions:
        wsi_files.extend(list(wsi_dir.rglob(f"*{ext}")))
        wsi_files.extend(list(wsi_dir.rglob(f"*{ext.upper()}")))
    
    wsi_files = sorted(set(wsi_files))  # Remove duplicates and sort
    
    if len(wsi_files) == 0:
        raise ValueError(f"No WSI files found in {wsi_dir} with extensions {wsi_extensions}")
    
    if verbose:
        print(f"Found {len(wsi_files)} WSI files")
    
    # Step 2: Load label mapping if provided
    label_mapping = {}
    if label_file:
        label_mapping = _load_label_file(label_file, verbose)
    
    # Step 3: Find annotations
    annotation_mapping = {}
    if annotation_dir and annotation_dir.exists():
        annotation_mapping = _build_annotation_mapping(annotation_dir, verbose)
    elif verbose and annotation_dir:
        print(f"Warning: Annotation directory not found: {annotation_dir}")
    
    # Step 4: Process each WSI and extract metadata
    metadata_list = []
    
    iterator = tqdm(wsi_files, desc="Processing WSIs") if verbose else wsi_files
    
    for wsi_path in iterator:
        try:
            metadata = _extract_wsi_metadata(
                wsi_path=wsi_path,
                annotation_mapping=annotation_mapping,
                label_mapping=label_mapping,
                verbose=False  # Suppress individual file messages
            )
            metadata_list.append(metadata)
        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to process {wsi_path.name}: {e}")
            # Still add with partial information
            metadata_list.append({
                'slide_id': _extract_slide_id(wsi_path.name),
                'filename': wsi_path.name,
                'wsi_path': str(wsi_path.absolute()),
                'label': None,
                'width': None,
                'height': None,
                'level_count': None,
                'annotation_path': None,
                'file_size_mb': wsi_path.stat().st_size / (1024**2) if wsi_path.exists() else None,
                'valid': False
            })
    
    # Step 5: Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Step 6: Validate and clean
    if verbose:
        print(f"\nMetadata Summary:")
        print(f"  Total slides: {len(df)}")
        print(f"  Valid slides: {df['valid'].sum()}")
        print(f"  Slides with labels: {df['label'].notna().sum()}")
        print(f"  Slides with annotations: {df['annotation_path'].notna().sum()}")
        
        if df['label'].notna().any():
            print(f"\nLabel distribution:")
            print(df['label'].value_counts().to_string())
    
    # Step 7: Export to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✅ Metadata CSV saved to: {output_path.absolute()}")
    
    return df


def _extract_slide_id(filename):
    """
    Extract slide ID from filename.
    Removes file extension and common prefixes/suffixes.
    
    Examples:
        'slide_001.svs' -> 'slide_001'
        'HER2_patient_123_A.tif' -> 'HER2_patient_123_A'
        'TCGA-XX-XXXX-01Z-00-DX1.svs' -> 'TCGA-XX-XXXX-01Z-00-DX1'
    """
    # Remove extension
    slide_id = Path(filename).stem
    
    # Remove common prefixes (optional, can be customized)
    # slide_id = re.sub(r'^(slide_|wsi_|image_)', '', slide_id, flags=re.IGNORECASE)
    
    return slide_id


def _extract_wsi_metadata(wsi_path, annotation_mapping, label_mapping, verbose=True):
    """Extract comprehensive metadata from a single WSI file."""
    
    slide_id = _extract_slide_id(wsi_path.name)
    
    metadata = {
        'slide_id': slide_id,
        'filename': wsi_path.name,
        'wsi_path': str(wsi_path.absolute()),
        'label': label_mapping.get(slide_id, None),
        'width': None,
        'height': None,
        'level_count': None,
        'annotation_path': annotation_mapping.get(slide_id, None),
        'file_size_mb': round(wsi_path.stat().st_size / (1024**2), 2),
        'valid': False
    }
    
    # Try to open WSI and extract dimensions (if OpenSlide available)
    if OPENSLIDE_AVAILABLE:
        try:
            slide = openslide.open_slide(str(wsi_path))
            metadata['width'] = slide.dimensions[0]
            metadata['height'] = slide.dimensions[1]
            metadata['level_count'] = slide.level_count
            metadata['valid'] = True
            slide.close()
        except Exception as e:
            if verbose:
                print(f"Warning: Could not open {wsi_path.name}: {e}")
            metadata['valid'] = False
    else:
        # Mark as valid by default if OpenSlide not available
        metadata['valid'] = True
    
    return metadata


def _load_label_file(label_file, verbose=True):
    """
    Load label mapping from CSV or text file.
    
    Expected format:
        - CSV: slide_id,label
        - TXT: slide_id<space or tab>label
    
    Returns:
        dict: {slide_id: label}
    """
    label_file = Path(label_file)
    
    if not label_file.exists():
        if verbose:
            print(f"Warning: Label file not found: {label_file}")
        return {}
    
    label_mapping = {}
    
    try:
        # Try CSV first
        if label_file.suffix.lower() in ['.csv']:
            df = pd.read_csv(label_file)
            # Support multiple column name conventions
            id_col = None
            label_col = None
            
            for col in df.columns:
                if col.lower() in ['slide_id', 'id', 'filename', 'slide', 'name']:
                    id_col = col
                if col.lower() in ['label', 'class', 'grade', 'score', 'her2']:
                    label_col = col
            
            if id_col and label_col:
                for _, row in df.iterrows():
                    slide_id = _extract_slide_id(str(row[id_col]))
                    label_mapping[slide_id] = row[label_col]
            else:
                raise ValueError(f"Could not identify slide_id and label columns in {label_file}")
        
        # Try plain text
        else:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Split by comma, tab, or space
                        parts = re.split(r'[,\t\s]+', line, maxsplit=1)
                        if len(parts) == 2:
                            slide_id = _extract_slide_id(parts[0])
                            label_mapping[slide_id] = parts[1]
        
        if verbose:
            print(f"Loaded {len(label_mapping)} labels from {label_file}")
    
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to load label file {label_file}: {e}")
    
    return label_mapping


def _build_annotation_mapping(annotation_dir, verbose=True):
    """
    Build mapping from slide_id to annotation file paths.
    Supports XML, JSON, TXT, and mask files.
    
    Returns:
        dict: {slide_id: annotation_path}
    """
    annotation_dir = Path(annotation_dir)
    
    # Common annotation file extensions
    annotation_extensions = ['.xml', '.json', '.txt', '.png', '.tif', '.tiff', '.npy']
    
    annotation_files = []
    for ext in annotation_extensions:
        annotation_files.extend(list(annotation_dir.rglob(f"*{ext}")))
        annotation_files.extend(list(annotation_dir.rglob(f"*{ext.upper()}")))
    
    annotation_mapping = {}
    
    for ann_path in annotation_files:
        slide_id = _extract_slide_id(ann_path.name)
        annotation_mapping[slide_id] = str(ann_path.absolute())
    
    if verbose and len(annotation_mapping) > 0:
        print(f"Found {len(annotation_mapping)} annotation files")
    
    return annotation_mapping


# Example usage and CLI support
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate metadata CSV from WSI files and annotations"
    )
    parser.add_argument(
        "wsi_dir",
        type=str,
        help="Directory containing whole slide images"
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        help="Directory containing annotation files (optional)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="metadata.csv",
        help="Output CSV file path (default: metadata.csv)"
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help="CSV/TXT file containing slide IDs and labels (optional)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="WSI file extensions to search for (default: .svs .tif .ndpi .mrxs)"
    )
    
    args = parser.parse_args()
    
    generate_metadata(
        wsi_dir=args.wsi_dir,
        annotation_dir=args.annotation_dir,
        output_csv=args.output,
        label_file=args.label_file,
        wsi_extensions=args.extensions,
        verbose=True
    )