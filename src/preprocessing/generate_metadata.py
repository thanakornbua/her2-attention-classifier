from __future__ import annotations

from pathlib import Path
import csv
from typing import List, Dict, Optional, Union
from tqdm import tqdm


def _gather_files(dir_path: Path, exts: List[str]) -> List[Path]:
    """Find all files with given extensions in directory."""
    if not dir_path.exists():
        return []
    files = []
    for ext in exts:
        files += list(dir_path.rglob(ext))
    seen = set()
    unique = []
    for p in files:
        ap = p.resolve()
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return sorted(unique)


def _index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    """Create a dictionary mapping filename stems to paths."""
    return {p.stem: p for p in paths}


def discover_for_source(base_dir: Path, source: str) -> List[dict]:
    """Discover WSI and annotation pairs for a single source."""
    source_dir = base_dir / source
    svs_dir = source_dir / "SVS"
    ann_dir = next((d for d in [source_dir / "Annotations", source_dir / "Annotation"] if d.exists()), source_dir / "Annotations")

    wsi_files = _gather_files(svs_dir, ["*.svs", "*.SVS"])
    ann_files = _gather_files(ann_dir, ["*.xml", "*.XML"])

    wsi_by_stem = _index_by_stem(wsi_files)
    ann_by_stem = _index_by_stem(ann_files)

    rows = []
    for stem, wsi_path in tqdm(wsi_by_stem.items(), desc=f"Processing {source}", leave=False):
        ann_path = ann_by_stem.get(stem)
        rows.append({
            "path": str(wsi_path),
            "slide_id": stem,
            "slide_name": wsi_path.name,
            "annotation_name": ann_path.name if ann_path else "",
            "annotation_dir": str(ann_path.parent) if ann_path else "",
        })

    return rows


def write_csv(rows: List[dict], out_path: Path, base_dir: Optional[Path] = None) -> None:
    """    
    Args:
        rows: List of dictionaries with slide information
        out_path: Output CSV file path
        base_dir: If provided, convert absolute paths to be relative to this directory
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "slide_id", "slide_name", "annotation_name", "annotation_dir"]
    
    # Convert absolute paths to relative if base_dir is provided
    if base_dir:
        base_dir = base_dir.resolve()
        processed_rows = []
        for r in tqdm(rows, desc="Converting paths", leave=False):
            row_copy = r.copy()
            for key in ["path", "annotation_dir"]:
                if row_copy.get(key):
                    try:
                        row_copy[key] = str(Path(row_copy[key]).relative_to(base_dir))
                    except ValueError:
                        pass
            processed_rows.append(row_copy)
        rows = processed_rows
    
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def discover_wsi(base_dir: Union[str, Path], sources: Union[str, List[str]],
                 output_path: Union[str, Path] = "outputs/wsi_index.csv",
                 relative_paths: bool = True) -> Path:
    
    base_dir = Path(base_dir).expanduser().resolve()
    
    if isinstance(sources, str):
        sources = [sources]
    
    all_rows = []
    for src in tqdm(sources, desc="Processing sources"):
        rows = discover_for_source(base_dir, src)
        all_rows.extend(rows)
    
    relative_base = Path.cwd() if relative_paths else None
    
    output_path_obj = Path(output_path)
    write_csv(all_rows, output_path_obj, base_dir=relative_base)
    print(f"✓ Saved {len(all_rows)} slides to {output_path}")
    return output_path_obj.resolve()
