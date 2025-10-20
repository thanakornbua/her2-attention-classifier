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
from IPython import get_ipython

# Enable autoreload
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

def extract_files(source_dir, svs_output_dir, log_output_dir):
    # Ensure output directories exist
    os.makedirs(svs_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    log_file = input("Enter the log file extension (e.g. .log): ")
    extracted = input("Enter the extracted file extension (e.g. SVS): ")

    # Count total files for progress tracking
    total_files = sum(len(files) for _, _, files in os.walk(source_dir))

    # Traverse the directory with total progress tracking
    with tqdm(total=total_files, desc="Processing all files") as pbar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(f'.{extracted}'):
                    # Move .svs files to svs_output_dir
                    shutil.copy(file_path, svs_output_dir)
                elif file == file.endswith(log_file):
                    # Move the log file to log_output_dir
                    shutil.copy(file_path, log_output_dir)
                pbar.update(1)
                # Delete the file after moving
                os.remove(file_path)

            for dir_name in dirs:
                if dir_name == log_file:
                    log_dir_path = os.path.join(root, dir_name)
                    # Move the entire log folder to log_output_dir
                    shutil.move(log_dir_path, log_output_dir)
                    pbar.update(len(os.listdir(log_dir_path)))

            # Remove the folder if it is empty
            if not os.listdir(root):
                os.rmdir(root)

if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Extract .svs files and move 'log' folders from a directory.")
    parser.add_argument("source_dir", help="Path to the source directory.")
    parser.add_argument("{extracted}", help="Path to the output directory for .svs files.")
    parser.add_argument("{log_file}_output_dir", help="Path to the output directory for 'log' folders.")
    args = parser.parse_args()

    extract_files(args.source_dir, args.extracted, args.log_file_output_dir)
