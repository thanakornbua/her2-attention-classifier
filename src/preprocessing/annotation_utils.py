import os
from glob import glob
from pathlib import Path

def resolve_annotation_path(annotation_path, wsi_path, base_dir='data'):
    """Resolve an annotation XML path given the CSV value and the wsi path.

    Returns an absolute path to the annotation XML if found, otherwise None.

    Behavior:
    - Accepts pandas NA-like values (will be treated as None).
    - If annotation_path is provided and not absolute, tries absolute path and
      joins with base_dir.
    - If missing, does a LIMITED search in annotation directories (not full recursive).
    """
    # Normalize base_dir
    if base_dir is None:
        base_dir = 'data'
    
    # Normalize NA-like values (pandas may pass NA objects)
    if annotation_path is not None:
        try:
            # avoid requiring pandas at import-time, only try when available
            import pandas as _pd
            if _pd.isna(annotation_path):
                annotation_path = None
        except Exception:
            pass

    if annotation_path is not None:
        annotation_path = str(annotation_path).strip()
        if annotation_path == '':
            annotation_path = None
    
    # If a path was provided, try to resolve it to an existing absolute file
    if annotation_path:
        if not os.path.isabs(annotation_path):
            candidate = os.path.abspath(annotation_path)
            if os.path.exists(candidate):
                return candidate
            candidate2 = os.path.join(base_dir, annotation_path)
            if os.path.exists(candidate2):
                return candidate2
        else:
            if os.path.exists(annotation_path):
                return annotation_path
    
    # If annotation_path was None/empty, do a LIMITED search
    # Infer source directory from wsi_path to avoid searching entire base_dir
    base_filename = os.path.splitext(os.path.basename(wsi_path))[0]
    
    # Try to determine source directory from wsi_path
    wsi_path_obj = Path(wsi_path)
    if wsi_path_obj.is_absolute():
        # Try to find parent directory that looks like a source (e.g., Yale_HER2_cohort)
        for parent in wsi_path_obj.parents:
            if parent.name in ['Yale_HER2_cohort', 'Yale_trastuzumab_response_cohort', 'TCGA_BRCA_Filtered']:
                # Search only within this source directory
                for ann_dir in ['Annotations', 'Annotation']:
                    check_path = parent / ann_dir / f'{base_filename}.xml'
                    if check_path.exists():
                        return str(check_path)
                return None
    
    # Fallback: check common annotation directory patterns (non-recursive)
    for ann_dir in ['Annotations', 'Annotation']:
        check_path = os.path.join(base_dir, ann_dir, f'{base_filename}.xml')
        if os.path.exists(check_path):
            return check_path
    
    return None
