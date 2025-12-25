import pandas as pd
from pathlib import Path
import openpyxl
from src.dataloader.discover_wsi import discover_wsi_paths
from src.dataloader.discover_xml import discover_xml_paths
from src.pipeline.logging_utils import setup_logging

logger = setup_logging("discovery")

def get_tcga_her2_status(case_name: str, metadata_path: str = 'data/TCGA_BRCA_Filtered/case&annotation_counts_clean.xlsx') -> int:
    """
    Extract HER2 status from TCGA metadata Excel file.
    
    Args:
        case_name (str): TCGA case ID (e.g., 'TCGA-A1-A0SP')
        metadata_path (str): Path to the metadata Excel file.
    
    Returns:
        int: 1 for Positive, 0 for Negative, 1 as default
    """
    try:
        wb = openpyxl.load_workbook(metadata_path)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] == case_name:
                if row[1] == 'Positive':
                    return 1
                elif row[1] == 'Negative':
                    return 0
                else:
                    return 1
    except Exception as e:
        logger.warning(f"Could not read TCGA metadata for {case_name}: {e}")
        pass
    return 1  # Default to positive if not found

def discover_data(data_root: str = 'data') -> pd.DataFrame:
    """
    Discovers WSI and XML files, merges them, and assigns labels.
    """
    logger.info(f"Discovering WSI files in {data_root}...")
    wsi_df = discover_wsi_paths(data_root, ('.svs',))
    if wsi_df is None or wsi_df.empty:
        raise FileNotFoundError(f"No WSI files found in {data_root}")
    
    wsi_df['case_name'] = wsi_df['file_name'].str.split('.').str[0]
    
    logger.info(f"Discovering XML files in {data_root}...")
    annotation_df = discover_xml_paths(data_root, ('.xml',))
    if annotation_df is not None and not annotation_df.empty:
        annotation_df['case_name'] = annotation_df['file_name'].str.split('.').str[0]
        
        # Merge
        paths_df = pd.merge(wsi_df, annotation_df, on='case_name', how='left', suffixes=('', '_annotation'))
    else:
        logger.warning("No XML annotations found. Proceeding with WSI only.")
        paths_df = wsi_df.copy()
        paths_df['full_path_annotation'] = None

    # Assign labels
    paths_df['label'] = 0
    
    # Yale HER2 Cohort
    paths_df.loc[paths_df['full_path'].astype(str).str.contains('Yale_HER2_cohort'), 'label'] = 1
    
    # Yale Trastuzumab Response Cohort
    paths_df.loc[paths_df['full_path'].astype(str).str.contains('Yale_trastuzumab_response_cohort'), 'label'] = 1
    
    # TCGA
    tcga_mask = paths_df['full_path'].astype(str).str.contains('TCGA_BRCA_Filtered')
    if tcga_mask.any():
        logger.info("Assigning TCGA labels...")
        paths_df.loc[tcga_mask, 'label'] = paths_df.loc[tcga_mask, 'case_name'].apply(get_tcga_her2_status)
        
    logger.info(f"Total slides: {len(paths_df)}")
    logger.info(f"Positive labels: {paths_df['label'].sum()}")
    logger.info(f"Negative labels: {len(paths_df) - paths_df['label'].sum()}")
    
    return paths_df
