import xml.etree.ElementTree as ET
import numpy as np
import cv2
import openslide
from multiprocessing import Pool
from tqdm import tqdm

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for annotation in root.findall('.//Annotation'):
        label = annotation.get("Name", "Unknown")
        for region in annotation.findall('.//Region'):
            vertices = []
            for vertex in region.findall('.//Vertex'):
                x = int(float(vertex.get("X")))
                y = int(float(vertex.get("Y")))
                vertices.append((x, y))
            if len(vertices) > 2:
                annotations.append({"label": label, "polygon": vertices})
    return annotations

def generate_mask(args):
    xml_path, wsi_path, output_path = args

    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.level_dimensions[0]  
    mask = np.zeros((h, w), dtype=np.uint8)

    polygons = parse_xml(xml_path)
    for item in polygons:
        poly = np.array(item["polygon"], dtype=np.int32)
        cv2.fillPoly(mask, [poly], color=255)

    cv2.imwrite(output_path, mask)
    return output_path

def process_all(xml_list, wsi_list, out_list, num_workers=4):
    args = list(zip(xml_list, wsi_list, out_list))
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(generate_mask, args), total=len(args), desc="Generating masks"):
            pass

# Example usage:
# xml_list = [...]  # paths to XML files
# wsi_list = [...]  # matching .svs files
# out_list = [...]  # output .png masks
# process_all(xml_list, wsi_list, out_list, num_workers=8)
