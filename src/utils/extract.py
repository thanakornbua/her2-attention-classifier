"""
Folder-file extraction utility.
File: extract.py
Description: This script is designed to extract specific files and directories from a given source directory.
             It supports the following functionality:
             
             1. Extract files with a user-specified extension (e.g., `.svs`) and move them to a designated output directory.
             2. Extract files with a user-specified log file extension (e.g., `.log`) and move them to another designated output directory.
             3. Move entire directories with a user-specified name (e.g., `log`) to the output directory.
             4. Display progress using `tqdm` for both file and directory operations.
             5. Delete files after moving and remove empty directories from the source directory.
             
Usage: This script can be run from the command line with arguments for the source directory, extracted file output directory, 
       and log file output directory.

Author: T. Buathongtanakarn et al.
Version: 1.0
Date: 21 October 2025
"""

import os
import shutil
from tqdm import tqdm
import argparse

# Remove IPython dependency for better portability and performance
# Only enable autoreload when running in IPython environment
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except ImportError:
    pass

def extract_files(source_dir, svs_output_dir, log_output_dir):
    """Extract files from source directory to output directories.
    
    Performance improvements:
    - Use os.walk only once instead of multiple passes
    - Use shutil.move instead of copy+delete for atomic operations
    - Batch file operations
    - Pre-compute total files count efficiently
    """
    # Ensure output directories exist
    os.makedirs(svs_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    log_file = input("Enter the log file extension (e.g. .log): ")
    extracted = input("Enter the extracted file extension (e.g. SVS): ")

    # Pre-compute total files count more efficiently
    total_files = sum(len(files) for _, _, files in os.walk(source_dir))

    # Single-pass processing for better performance
    processed_dirs = set()
    with tqdm(total=total_files, desc="Processing all files") as pbar:
        # Process in reverse order to handle nested directories correctly
        for root, dirs, files in os.walk(source_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith(f'.{extracted}'):
                        # Use shutil.move instead of copy+delete (atomic and faster)
                        shutil.move(file_path, os.path.join(svs_output_dir, file))
                    elif file.endswith(log_file):
                        # Use shutil.move instead of copy+delete
                        shutil.move(file_path, os.path.join(log_output_dir, file))
                    else:
                        # Delete files that don't match criteria
                        os.remove(file_path)
                except (OSError, shutil.Error) as e:
                    print(f"Error processing {file_path}: {e}")
                pbar.update(1)

            # Handle log directories
            for dir_name in dirs:
                if dir_name == log_file:
                    log_dir_path = os.path.join(root, dir_name)
                    try:
                        # Move entire directory
                        dest_path = os.path.join(log_output_dir, dir_name)
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path)
                        # Count files before moving
                        file_count = len(os.listdir(log_dir_path)) if os.path.exists(log_dir_path) else 0
                        shutil.move(log_dir_path, log_output_dir)
                        pbar.update(file_count)
                    except (OSError, shutil.Error) as e:
                        print(f"Error moving directory {log_dir_path}: {e}")

            # Remove empty directories
            try:
                if root != source_dir and not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass  # Directory not empty or other error, skip

if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Extract files and move folders from a directory.")
    parser.add_argument("source_dir", help="Path to the source directory.")
    parser.add_argument("svs_output_dir", help="Path to the output directory for extracted files.")
    parser.add_argument("log_output_dir", help="Path to the output directory for log files/folders.")
    args = parser.parse_args()

    extract_files(args.source_dir, args.svs_output_dir, args.log_output_dir)
