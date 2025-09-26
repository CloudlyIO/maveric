# export_metrics.py
# Export function for Load Balancing module metrics table
# Provides functionality to export recommended tilt configurations to various formats

import logging
import os
import pandas as pd
from typing import Dict, Any, List
import datetime
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_metrics(
    metrics: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Exports a DataFrame of Load Balancing metrics (recommended tilt configurations) to various file formats.
    
    The metrics DataFrame should contain the following columns based on rl_predictor.py structure:
    - cell_id: Unique identifier for each cell
    - predicted_cell_el_deg: Predicted electrical tilt angle (0-20 degrees)
    
    Args:
        metrics (pd.DataFrame): The DataFrame containing the Load Balancing metrics to export.
                                Expected columns: ['cell_id', 'predicted_cell_el_deg']
        output_path (str): The base path for the output file(s).
                          E.g., "./exported_lb_metrics" will result in "./exported_lb_metrics.csv", etc.
        format (str): The desired export format. Can be 'csv', 'json', 'excel', or 'all'.
                      Defaults to 'csv'.
        include_metadata (bool): Whether to include metadata (timestamp, format, description) in JSON output.
    
    Returns:
        Dict[str, Any]: A dictionary indicating success status and files created.
                        Example: {'success': True, 'files_created': ['path/to/file.csv']}
    
    Example:
        >>> import pandas as pd
        >>> from export_func import export_metrics
        >>> 
        >>> # Sample metrics data from load_balancing rl_predictor.py
        >>> metrics_df = pd.DataFrame({
        ...     'cell_id': ['cell_001', 'cell_002', 'cell_003'],
        ...     'predicted_cell_el_deg': [8.0, 12.0, 5.0]
        ... })
        >>> 
        >>> # Export to CSV
        >>> result = export_metrics(metrics_df, "./lb_metrics", format="csv")
        >>> print(f"Success: {result['success']}")
        >>> print(f"Files: {result['files_created']}")
    """
    logger.info("Starting Load Balancing metrics export")
    result = {'success': False, 'files_created': [], 'errors': []}
    
    # Validate input
    if metrics.empty:
        result['errors'].append("Metrics DataFrame is empty, nothing to export.")
        logger.warning("Metrics DataFrame is empty, nothing to export.")
        return result
    
    # Validate required columns
    required_columns = ['cell_id', 'predicted_cell_el_deg']
    missing_columns = [col for col in required_columns if col not in metrics.columns]
    if missing_columns:
        result['errors'].append(f"Missing required columns: {missing_columns}")
        logger.error(f"Missing required columns: {missing_columns}")
        return result
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{output_path}_{timestamp}"
    
    # Determine export formats
    export_formats = []
    if format == 'all':
        export_formats = ['csv', 'json', 'xlsx']
    elif format in ['csv', 'json', 'excel', 'xlsx']:
        export_formats = [format if format != 'excel' else 'xlsx']
    else:
        result['errors'].append(f"Unsupported export format: {format}")
        logger.error(f"Unsupported export format: {format}")
        return result
    
    logger.info(f"Preparing to export {len(metrics)} metrics records to {len(export_formats)} format(s)")
    
    # Export to each format
    for fmt in export_formats:
        try:
            file_path = f"{base_filename}.{fmt}"
            
            if fmt == 'csv':
                # Export to CSV format
                metrics.to_csv(file_path, index=False)
                logger.info(f"Exported CSV to: {file_path}")
                
            elif fmt == 'json':
                # Export to JSON format with optional metadata
                json_data = metrics.to_dict(orient='records')
                
                if include_metadata:
                    metadata = {
                        "timestamp": timestamp,
                        "format": "json",
                        "description": "Load Balancing Predicted Optimal Tilt Configuration",
                        "module": "Load Balancing (CCO)",
                        "metrics_count": len(metrics),
                        "columns": list(metrics.columns),
                        "export_timestamp": datetime.datetime.now().isoformat(),
                        "tilt_range": "0-20 degrees",
                        "optimization_type": "Coverage and Capacity Optimization (CCO)"
                    }
                    json_output = {
                        "metadata": metadata,
                        "data": json_data
                    }
                else:
                    json_output = json_data
                
                with open(file_path, 'w') as f:
                    json.dump(json_output, f, indent=2)
                logger.info(f"Exported JSON to: {file_path}")
                
            elif fmt == 'xlsx':
                # Export to Excel format
                metrics.to_excel(file_path, index=False, sheet_name='LB_Metrics')
                logger.info(f"Exported Excel to: {file_path}")
                
            else:
                raise ValueError(f"Unhandled format: {fmt}")
            
            result['files_created'].append(file_path)
            
        except Exception as e:
            error_msg = f"Failed to export to {fmt}: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
    
    # Determine final success status
    if not result['errors']:
        result['success'] = True
        logger.info(f"Export completed successfully. Files created: {len(result['files_created'])}")
    else:
        logger.error(f"Export completed with errors. Errors: {result['errors']}")
    
    return result


def validate_metrics_format(metrics: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates that the metrics DataFrame has the expected structure for Load Balancing module.
    
    Args:
        metrics (pd.DataFrame): The DataFrame to validate
    
    Returns:
        Dict[str, Any]: Validation result with success status and any errors
    """
    result = {'valid': True, 'errors': [], 'warnings': []}
    
    if metrics.empty:
        result['valid'] = False
        result['errors'].append("DataFrame is empty")
        return result
    
    # Check required columns
    required_columns = ['cell_id', 'predicted_cell_el_deg']
    missing_columns = [col for col in required_columns if col not in metrics.columns]
    if missing_columns:
        result['valid'] = False
        result['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check data types and values
    if 'predicted_cell_el_deg' in metrics.columns:
        # Check for numeric values within expected range (0-20 degrees)
        numeric_mask = pd.to_numeric(metrics['predicted_cell_el_deg'], errors='coerce').notna()
        if not numeric_mask.all():
            invalid_values = metrics[~numeric_mask]['predicted_cell_el_deg'].unique()
            result['warnings'].append(f"Found non-numeric predicted_cell_el_deg values: {invalid_values}")
        
        # Check tilt range (0-20 degrees as per TILT_SET_PREDICTOR)
        numeric_tilts = pd.to_numeric(metrics['predicted_cell_el_deg'], errors='coerce')
        out_of_range = (numeric_tilts < 0) | (numeric_tilts > 20)
        if out_of_range.any():
            out_of_range_values = metrics[out_of_range]['predicted_cell_el_deg'].unique()
            result['warnings'].append(f"Found tilt values outside expected range (0-20): {out_of_range_values}")
    
    # Check cell_id format
    if 'cell_id' in metrics.columns:
        if metrics['cell_id'].isna().any():
            result['warnings'].append("Found null values in cell_id column")
        
        # Check for duplicate cell_ids
        if metrics['cell_id'].duplicated().any():
            duplicate_cells = metrics[metrics['cell_id'].duplicated()]['cell_id'].unique()
            result['warnings'].append(f"Found duplicate cell_id values: {duplicate_cells}")
    
    return result
