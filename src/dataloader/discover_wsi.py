"""
    Discovers Whole Slide Image (WSI) files in a directory and its subdirectories,
    saves their full paths and file names to a CSV file, and returns a DataFrame.

    This function is designed to be called from a script or a Jupyter Notebook.

    Args:
        root_dir (str): The root directory to start the search from.
        wsi_formats (Tuple[str, ...], optional): A tuple of WSI file extensions
            to search for. The search is case-insensitive.
            Defaults to ('.svs', '.ndpi', '.tif', '.tiff').

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing 'full_path' and 'file_name'
        for each discovered WSI file, or None if no files are found.

    Authors:
        T. Buathongtanakarn (2025),
        P. Sirithipvanich (2025),
        et al.

    LLM-Assistant:
        1. Gemini 2.5 Pro
    
    With help of:
        1. Multiple stackoverflow articles on file system traversal and pandas DataFrame creation.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import gc

def discover_wsi_paths(
    root_dir: str,
    wsi_formats: Tuple[str, ...] = ('.svs', '.ndpi', '.tif', '.tiff')
) -> Optional[pd.DataFrame]:
    wsi_data: List[dict] = []
    root_path = Path(root_dir).resolve()

    if not root_path.is_dir():
        raise FileNotFoundError(f"Error: Root directory not found at '{root_dir}'")

    print(f"Searching for files with extensions {wsi_formats} in '{root_dir}'...")

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(wsi_formats):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(root_path)
                wsi_data.append({
                    'full_path': str(Path(root_dir) / relative_path),
                    'file_name': file
                })

    if not wsi_data:
        print("No WSI files found.")
        gc.collect()
        return None

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(wsi_data)
    print(f"Found {len(wsi_data)} WSI files.")
    
    gc.collect()
    return df