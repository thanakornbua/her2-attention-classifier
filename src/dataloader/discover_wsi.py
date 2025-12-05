"""
    Discovers Whole Slide Image (WSI) files in a directory and its subdirectories,
    saves their relative paths and file names to a CSV file, and returns a DataFrame.

    This function is designed to be called from a script or a Jupyter Notebook.

    Args:
        root_dir (str): The root directory to start the search from.
        output_csv_path (str): The path to save the output CSV file.
        wsi_formats (Tuple[str, ...], optional): A tuple of WSI file extensions
            to search for. The search is case-insensitive.
            Defaults to ('.svs', '.ndpi', '.tif', '.tiff').

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing 'relative_path' and 'file_name'
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
    output_csv_path: str,
    wsi_formats: Tuple[str, ...] = ('.svs', '.ndpi', '.tif', '.tiff')
) -> Optional[pd.DataFrame]:
    wsi_data: List[dict] = []
    root_path = Path(root_dir)

    if not root_path.is_dir():
        raise FileNotFoundError(f"Error: Root directory not found at '{root_dir}'")

    print(f"Searching for files with extensions {wsi_formats} in '{root_dir}'...")

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(wsi_formats):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(root_path)
                wsi_data.append({
                    'relative_path': str(relative_path),
                    'file_name': file
                })

    if not wsi_data:
        print("No WSI files found.")
        # Create an empty DataFrame with specified columns and save it
        df = pd.DataFrame(columns=['relative_path', 'file_name'])
        output_dir = Path(output_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Empty CSV created at '{output_csv_path}'.")
        gc.collect()
        return df

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(wsi_data)
    
    # Ensure the output directory exists
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"Found {len(wsi_data)} WSI files. Paths and names saved to '{output_csv_path}'.")
    
    gc.collect()
    return df