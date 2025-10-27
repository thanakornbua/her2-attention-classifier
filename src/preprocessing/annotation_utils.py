import os
from glob import glob

def resolve_annotation_path(annotation_path, wsi_path, base_dir='data'):
    """Resolve an annotation XML path given the CSV value and the wsi path.

    Returns an absolute path to the annotation XML if found, otherwise None.

    Behavior:
    - Accepts pandas NA-like values (will be treated as None).
    - If annotation_path is provided and not absolute, tries absolute path and
      joins with base_dir.
    - If missing, searches for common patterns under base_dir using the
      wsi basename: <base>.xml or <base>_*.xml.
    """
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

    # Fallback: search by slide basename under base_dir (recursive)
    base = os.path.splitext(os.path.basename(wsi_path))[0]
    patterns = [
        os.path.join(base_dir, '**', f'{base}.xml'),
        os.path.join(base_dir, '**', f'{base}_*.xml'),
        f'**/{base}.xml',
    ]
    found = []
    for p in patterns:
        try:
            found.extend(glob(p, recursive=True))
        except Exception:
            # ignore malformed glob patterns
            pass
    found = sorted(set(found))
    if found:
        return found[0]

    return None
