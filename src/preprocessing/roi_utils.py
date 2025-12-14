"""
ROI (Region of Interest) utilities for parsing XML annotations and spatial filtering.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def parse_asap_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Parse ASAP/Aperio XML annotation file and extract ROI polygons.
    Handles both standard ASAP format and malformed XML with recovery.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of dicts with 'label' and 'polygon' keys.
        Each polygon is a list of (x, y) coordinate tuples.
    """
    polygons = []
    
    try:
        # Try lxml parser first (more robust)
        if hasattr(ET, 'parse'):
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        else:
            # Fallback to standard ElementTree
            import xml.etree.ElementTree as ET_std
            tree = ET_std.parse(str(xml_path))
            root = tree.getroot()
        
        # Try ASAP format: Annotations -> Annotation -> Coordinates -> Coordinate
        annotations = root.find('Annotations')
        if annotations is not None:
            for annotation in annotations.findall('Annotation'):
                label = annotation.get('Name') or annotation.get('Value') or 'Unknown'
                coordinates = annotation.find('Coordinates')
                if coordinates is not None:
                    points = []
                    for coordinate in coordinates.findall('Coordinate'):
                        try:
                            x = int(float(coordinate.get('X', 0)))
                            y = int(float(coordinate.get('Y', 0)))
                            points.append((x, y))
                        except (ValueError, TypeError):
                            continue
                    
                    if len(points) >= 3:  # Valid polygon needs at least 3 points
                        polygons.append({'label': label, 'polygon': points})
        
        # Try alternative format: Annotation -> Region -> Vertex (some ASAP variants)
        if not polygons:
            for annotation in root.findall('.//Annotation'):
                label = annotation.get('Name') or annotation.get('Value') or 'Unknown'
                for region in annotation.findall('.//Region'):
                    points = []
                    for vertex in region.findall('.//Vertex'):
                        try:
                            x = int(float(vertex.get('X', 0)))
                            y = int(float(vertex.get('Y', 0)))
                            points.append((x, y))
                        except (ValueError, TypeError):
                            continue
                    
                    if len(points) >= 3:
                        polygons.append({'label': label, 'polygon': points})
                
    except Exception as e:
        import logging
        logger = logging.getLogger('wsi-pipeline')
        logger.warning(f"Could not parse XML {xml_path}: {e}")
        
    return polygons


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    Works with list of (x, y) tuples or numpy arrays.
    
    Args:
        point: (x, y) coordinates
        polygon: List of (x, y) tuples or Nx2 array of polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0][0], polygon[0][1]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][0], polygon[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
        
    return inside


def patch_in_roi(x: int, y: int, patch_size: int, roi_polygons: List[Dict[str, Any]]) -> bool:
    """
    Check if a patch (defined by top-left corner and size) overlaps with any ROI polygon.
    Uses center point check for efficiency.
    
    Args:
        x: Patch top-left x coordinate
        y: Patch top-left y coordinate
        patch_size: Size of square patch
        roi_polygons: List of ROI dicts with 'polygon' key (from parse_asap_xml)
        
    Returns:
        True if patch center is within any ROI
    """
    if not roi_polygons:
        return True  # No ROIs defined, accept all patches
    
    # Check patch center point
    center_x = x + patch_size / 2
    center_y = y + patch_size / 2
    
    for roi_dict in roi_polygons:
        polygon = roi_dict.get('polygon', [])
        if polygon and point_in_polygon((center_x, center_y), polygon):
            return True
            
    return False


def get_annotation_path(wsi_path: str) -> str:
    """
    Get corresponding XML annotation path for a WSI file.
    
    Args:
        wsi_path: Path to WSI file
        
    Returns:
        Path to corresponding XML file, or empty string if not found
    """
    wsi_path = Path(wsi_path)
    
    # Try to find annotation in parallel Annotations directory
    parent_dir = wsi_path.parent.parent  # Go up from SVS folder
    annotations_dir = parent_dir / "Annotations"
    
    if not annotations_dir.exists():
        # Try sibling Annotations folder
        annotations_dir = wsi_path.parent.parent / "Annotations"
    
    if annotations_dir.exists():
        # Look for XML with matching case name
        case_name = wsi_path.stem.split('.')[0]
        xml_path = annotations_dir / f"{case_name}.xml"
        
        if xml_path.exists():
            return str(xml_path)
    
    return ""
