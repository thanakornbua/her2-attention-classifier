"""
    Discovers XML annotation files in a directory and its subdirectories,
    and returns a DataFrame with their full paths and file names.

    This function is designed to be called from a script or a Jupyter Notebook.

    Args:
        root_dir (str): The root directory to start the search from.
        xml_formats (Tuple[str, ...], optional): A tuple of XML file extensions
            to search for. The search is case-insensitive.
            Defaults to ('.xml',).

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing 'full_path' and 'file_name'
        for each discovered XML file, or None if no files are found.

    Authors:
        T. Buathongtanakarn (2025),
        P. Sirithipvanich (2025),
        et al.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import gc

def discover_xml_paths(
    root_dir: str,
    xml_formats: Tuple[str, ...] = ('.xml',)
) -> Optional[pd.DataFrame]:
    xml_data: List[dict] = []
    root_path = Path(root_dir).resolve()

    if not root_path.is_dir():
        raise FileNotFoundError(f"Error: Root directory not found at '{root_dir}'")

    print(f"Searching for files with extensions {xml_formats} in '{root_dir}'...")

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(xml_formats):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(root_path)
                xml_data.append({
                    'full_path': str(Path(root_dir) / relative_path),
                    'file_name': file,
                    'case_name': file.split('.')[0]  # Extract case name from filename
                })

    if not xml_data:
        print("No XML files found.")
        gc.collect()
        return None

    # Create a DataFrame
    df = pd.DataFrame(xml_data)
    print(f"Found {len(xml_data)} XML files.")
    
    gc.collect()
    return df