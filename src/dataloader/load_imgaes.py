# python
import os
from PIL import Image
import numpy as np

def load_images_from_directory(directory_path, recursive=True):
    """
    Loads all valid image files (PNG, JPG, JPEG) from a specified directory.
    When recursive is True, images inside subdirectories are also discovered.
    Converts images to RGB and returns them as NumPy arrays along with their
    paths relative to directory_path.

    Args:
        directory_path (str): The path to the directory containing image files.
        recursive (bool): If True, search subdirectories recursively.

    Returns:
        tuple: (list_of_images_as_numpy_uint8, list_of_relative_paths)
    """
    loaded_images = []
    filenames = []
    supported_extensions = ('.png', '.jpg', '.jpeg')

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}.")
        return [], []

    if recursive:
        walker = os.walk(directory_path)
    else:
        # emulate a single-level walk for non-recursive behavior
        walker = [(directory_path, [], os.listdir(directory_path))]

    for root, _, files in walker:
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    try:
                        img = Image.open(file_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img)
                        loaded_images.append(img_array)
                        rel_path = os.path.relpath(file_path, directory_path)
                        filenames.append(rel_path)
                    except Exception as e:
                        print(f"Warning: Could not load or process image {file_path}: {e}")

    if not loaded_images:
        print(f"No supported image files found in {directory_path}.")

    print(f"Loaded {len(loaded_images)} images from {directory_path}.")
    return loaded_images, filenames
