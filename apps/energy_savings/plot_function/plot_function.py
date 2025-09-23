# plot_function.py
"""
Energy Savings Plot Function

This module provides a Plot function that generates D3-compatible JSON payload for energy savings visualization.
The function loads an RL model, performs inference for a given tick, and uses BDT models for cell attachment logic.

"""

import logging
import os
import sys
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add project root to Python path to enable radp module imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(APP_DIR)))
sys.path.insert(0, PROJECT_ROOT)

# Import required libraries with error handling
try:
    from stable_baselines3 import PPO  # RL model for cell configuration optimization
    from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin  # BDT model for RF prediction
    from radp.digital_twin.utils import constants as c  # Constants for RF calculations
    from radp.digital_twin.utils.cell_selection import perform_attachment  # Cell attachment logic
except ImportError as e:
    print(f"FATAL: Error importing libraries: {e}")
    sys.exit(1)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Tilt configuration set used by the RL model
TILT_SET = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


def Plot(tick: int, input_dataset: pd.DataFrame, topology_path: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Generate D3-compatible JSON payload for energy savings visualization.
    
    This function:
    1. Loads the Energy Savings RL model
    2. Performs inference for the given tick to get optimal cell configurations
    3. Uses BDT models to run attachment logic and determine UE-cell associations
    4. Returns a structured JSON payload for D3 visualization
    
    Args:
        tick (int): Time tick (0-23) for which to generate the plot
        input_dataset (pd.DataFrame): UE dataset with columns ['loc_x', 'loc_y', 'mock_ue_id']
        topology_path (str, optional): Path to topology CSV file. If None, uses placeholder data
        config_path (str, optional): Path to config CSV file. If None, uses default values
        
    Returns:
        Dict[str, Any]: D3-compatible JSON payload with structure:
        {
            "groups": [
                {
                    "title": "cells",
                    "data": [{"x": lat, "y": lon}, ...]  # Cell tower locations
                },
                {
                    "title": "cell_id_1", 
                    "data": [{"x": lat, "y": lon}, ...]  # UEs served by cell_id_1
                },
                // ... more cell groups
            ]
        }
        
    Raises:
        FileNotFoundError: If model files are not found
        ValueError: If tick is out of range or data format is invalid
        Exception: For other processing errors
    """
    
    # Validate tick range (0-23 represents hours in a day)
    if not (0 <= tick <= 23):
        raise ValueError(f"Tick {tick} is out of range (0-23)")
    
    # Ensure input dataset is not empty
    if input_dataset.empty:
        raise ValueError("Input dataset is empty")
    
    # Validate required columns exist in the input dataset
    # These columns are essential for UE location and identification
    required_columns = ['loc_x', 'loc_y', 'mock_ue_id']
    missing_columns = [col for col in required_columns if col not in input_dataset.columns]
    if missing_columns:
        raise ValueError(f"Input dataset missing required columns: {missing_columns}")
    
    # Hardcoded model paths
    # These paths point to the model files in the parent energy_savings directory
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(APP_DIR)
    BDT_MODEL_PATH = os.path.join(PARENT_DIR, "bdt_model_map.pickle")
    RL_MODEL_PATH = os.path.join(PARENT_DIR, "energy_saver_agent.zip")

    # Load the trained RL model (PPO agent) for cell configuration optimization
    logger.info(f"Loading RL model from {RL_MODEL_PATH}")
    rl_model_file = RL_MODEL_PATH if RL_MODEL_PATH.endswith(".zip") else f"{RL_MODEL_PATH}.zip"
    if not os.path.exists(rl_model_file):
        raise FileNotFoundError(f"RL Model file not found: {rl_model_file}")
    
    # Load the PPO model using Stable Baselines3
    rl_model = PPO.load(rl_model_file)
    logger.info("RL model loaded successfully")
    
    # Load BDT model
    # Load the Bayesian Digital Twin model for RF signal prediction
    logger.info(f"Loading BDT model from {BDT_MODEL_PATH}")
    if not os.path.exists(BDT_MODEL_PATH):
        raise FileNotFoundError(f"BDT Model file not found: {BDT_MODEL_PATH}")
    
    # Load the BDT model map (contains models for each cell)
    bdt_model_map = BayesianDigitalTwin.load_model_map_from_pickle(BDT_MODEL_PATH)
    logger.info(f"Loaded BDT model map for {len(bdt_model_map)} cells")
    
    # Load topology and configuration data
    # Load cell topology data (cell tower locations and IDs)
    if topology_path and os.path.exists(topology_path):
        topology_df = pd.read_csv(topology_path)
        logger.info(f"Loaded topology data with {len(topology_df)} cells")
    else:
        # Use placeholder data if no topology file is provided
        logger.warning("No topology path provided or file not found, using placeholder data")
        topology_df = _create_placeholder_topology()
    
    # Load cell configuration data (initial cell states and parameters)
    if config_path and os.path.exists(config_path):
        config_df = pd.read_csv(config_path)
        logger.info(f"Loaded configuration data with {len(config_df)} cells")
    else:
        # Use default values if no config file is provided
        logger.warning("No config path provided or file not found, using default values")
        config_df = _create_default_config(topology_df)
    
    # Merge topology and configuration data into a single dataframe
    # This combines cell locations with their configuration parameters
    site_config_df = pd.merge(topology_df, config_df, on="cell_id", how="left")
    site_config_df["cell_el_deg"].fillna(TILT_SET[len(TILT_SET) // 2], inplace=True)
    
    # Add required columns with default values if missing
    # These columns are needed for RF calculations and cell operations
    required_cols = {"hTx": 25.0, "hRx": 1.5, "cell_az_deg": 0.0, "cell_carrier_freq_mhz": 2100.0}
    for col, val in required_cols.items():
        if col not in site_config_df.columns:
            site_config_df[col] = val
    
    # Perform RL inference to get optimal cell configurations for the given tick
    # This determines which cells should be ON/OFF and their optimal tilt angles
    logger.info(f"Running RL inference for tick {tick}")
    action_indices, _ = rl_model.predict(tick, deterministic=True)
    
    # Convert RL action indices to actual cell configurations
    # Each action index maps to a specific tilt angle or "OFF" state
    cell_configs = _create_cell_configurations_from_actions(action_indices, site_config_df)
    
    # Use BDT models to predict RF signal strength and perform cell attachment
    # This determines which UEs are served by which cells based on signal strength
    logger.info("Running BDT attachment logic")
    attached_data = _run_bdt_attachment(input_dataset, cell_configs, bdt_model_map)
    
    # Generate the final D3-compatible JSON payload for visualization
    logger.info("Generating D3-compatible JSON payload")
    plot_data = _generate_plot_payload(attached_data, cell_configs)
    
    logger.info(f"Successfully generated plot data for tick {tick}")
    return plot_data


def _create_cell_configurations_from_actions(action_indices: List[int], site_config_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cell configuration DataFrame from RL action indices using actual topology data.
    
    This function converts RL model action indices into actual cell configurations.
    Each action index corresponds to either a specific tilt angle or "OFF" state.
    
    Args:
        action_indices (List[int]): RL model action indices for each cell
        site_config_df (pd.DataFrame): Site configuration data with cell locations and properties
        
    Returns:
        pd.DataFrame: Cell configuration with columns ['cell_id', 'cell_lon', 'cell_lat', 'cell_el_deg', ...]
    """
    cell_configs = []
    
    # Get cell IDs in the same order as the RL model expects
    # This ensures the action indices correspond to the correct cells
    cell_ids_ordered = site_config_df['cell_id'].unique().tolist()
    
    # Validate that action indices match the number of cells
    if len(action_indices) != len(cell_ids_ordered):
        logger.error(f"Action length {len(action_indices)} != num cells {len(cell_ids_ordered)}")
        return pd.DataFrame()
    
    # Process each cell's action index
    for i, action_idx in enumerate(action_indices):
        cell_id = cell_ids_ordered[i]
        cell_data = site_config_df[site_config_df['cell_id'] == cell_id].iloc[0]
        
        # Check if cell should be ON (action_idx < len(TILT_SET)) or OFF
        if action_idx < len(TILT_SET):
            # Cell is ON with specific tilt angle from TILT_SET
            cell_config = cell_data.copy()
            cell_config['cell_el_deg'] = TILT_SET[action_idx]
            cell_configs.append(cell_config)
        # If action_idx == len(TILT_SET), cell is OFF (not included in config)
    
    return pd.DataFrame(cell_configs)


def _run_bdt_attachment(ue_data: pd.DataFrame, cell_configs: pd.DataFrame, bdt_model_map: Dict) -> pd.DataFrame:
    """
    Run BDT attachment logic to determine which UEs are served by which cells.
    
    This function uses BDT models to predict RF signal strength for each UE-cell pair
    and then performs cell attachment logic to assign UEs to their optimal serving cells.
    
    Args:
        ue_data (pd.DataFrame): UE dataset with location information
        cell_configs (pd.DataFrame): Active cell configurations from RL inference
        bdt_model_map (Dict): BDT model map for each cell
        
    Returns:
        pd.DataFrame: UE data with serving cell information
    """
    # Check if there are any active cells to serve UEs
    if cell_configs.empty:
        logger.warning("No active cells found, returning empty attachment data")
        return pd.DataFrame()
    
    # Initialize list to store RF predictions for all cells
    all_preds_list = []
    active_cell_ids = cell_configs['cell_id'].unique()
    
    # Process each active cell to generate RF predictions
    for cell_id in active_cell_ids:
        # Get the BDT predictor for this specific cell
        bdt_predictor = bdt_model_map.get(cell_id)
        if not bdt_predictor:
            logger.warning(f"No BDT predictor found for cell {cell_id}")
            continue
        
        # Get configuration data for this specific cell
        cell_cfg_df = cell_configs[cell_configs['cell_id'] == cell_id]
        
        # Create prediction frames for BDT model input
        # This prepares the data in the format expected by the BDT model
        pred_frames = BayesianDigitalTwin.create_prediction_frames(
            site_config_df=cell_cfg_df, 
            prediction_frame_template=ue_data
        )
        df_for_pred = pred_frames.get(cell_id)
        
        # Run BDT prediction if data is available
        if df_for_pred is not None and not df_for_pred.empty:
            # Use BDT model to predict RF signal strength for this cell
            bdt_predictor.predict_distributed_gpmodel(prediction_dfs=[df_for_pred])
            all_preds_list.append(df_for_pred)
    
    # Check if any predictions were generated
    if not all_preds_list:
        logger.warning("No BDT predictions generated")
        return pd.DataFrame()
    
    # Combine all predictions from different cells into a single dataframe
    combined_preds = pd.concat(all_preds_list, ignore_index=True)
    
    # Perform cell attachment logic to assign UEs to their optimal serving cells
    # This uses the RF predictions to determine which cell each UE should connect to
    return perform_attachment(combined_preds, cell_configs)


def _generate_plot_payload(attached_data: pd.DataFrame, cell_configs: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate D3-compatible JSON payload from attachment data.
    
    Args:
        attached_data (pd.DataFrame): UE data with serving cell information
        cell_configs (pd.DataFrame): Cell configuration data
        
    Returns:
        Dict[str, Any]: D3-compatible plot data structure
    """
    groups = []
    
    # Add cell tower locations group
    if not cell_configs.empty:
        cell_points = []
        for _, cell in cell_configs.iterrows():
            cell_points.append({
                "x": cell['cell_lon'],
                "y": cell['cell_lat']
            })
        
        groups.append({
            "title": "cells",
            "data": cell_points
        })
    
    # Add UE groups by serving cell
    if not attached_data.empty and 'serving_cell_id' in attached_data.columns:
        # Group UEs by their serving cell
        for cell_id, cell_ues in attached_data.groupby('serving_cell_id'):
            ue_points = []
            for _, ue in cell_ues.iterrows():
                ue_points.append({
                    "x": ue['loc_x'],
                    "y": ue['loc_y']
                })
            
            groups.append({
                "title": cell_id,
                "data": ue_points
            })
    
    return {
        "groups": groups
    }


def _create_placeholder_topology() -> pd.DataFrame:
    """
    Create placeholder topology data for testing purposes.
    
    This function generates dummy cell tower locations when no topology file is provided.
    It creates a 3x2 grid of cells for testing the plot function.
    
    Returns:
        pd.DataFrame: Placeholder topology with 5 cells
    """
    import numpy as np
    
    # Create 5 cells in a grid pattern
    # This provides a realistic test scenario with multiple cell towers
    cells = []
    for i in range(5):
        cell_id = f"cell_{i}"
        # Create a 3x2 grid pattern for cell locations
        cell_lon = 40.7 + (i % 3) * 0.01  # Longitude: 3 columns
        cell_lat = -74.0 + (i // 3) * 0.01  # Latitude: 2 rows
        cells.append({
            'cell_id': cell_id,
            'cell_lon': cell_lon,
            'cell_lat': cell_lat
        })
    
    return pd.DataFrame(cells)


def _create_default_config(topology_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create default configuration data for topology.
    
    This function generates default cell configuration parameters when no config file is provided.
    It uses reasonable default values for RF calculations and cell operations.
    
    Args:
        topology_df (pd.DataFrame): Topology data with cell IDs
        
    Returns:
        pd.DataFrame: Default configuration data for all cells
    """
    configs = []
    for _, cell in topology_df.iterrows():
        configs.append({
            'cell_id': cell['cell_id'],
            'cell_el_deg': TILT_SET[len(TILT_SET) // 2],  # Default tilt (middle of range)
            'hTx': 25.0,  # Transmitter height in meters
            'hRx': 1.5,   # Receiver height in meters
            'cell_az_deg': 0.0,  # Azimuth angle (degrees)
            'cell_carrier_freq_mhz': 2100.0  # Carrier frequency in MHz
        })
    
    return pd.DataFrame(configs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample UE data
    sample_ue_data = pd.DataFrame({
        'loc_x': np.random.uniform(40.7, 40.8, 100),
        'loc_y': np.random.uniform(-74.05, -73.95, 100),
        'mock_ue_id': [f'ue_{i}' for i in range(100)]
    })
    
    try:
        # Generate plot data
        plot_data = Plot(
            tick=12,
            input_dataset=sample_ue_data,
            topology_path=None,  # Will use placeholder data
            config_path=None     # Will use default values
        )
        
        print("Generated plot data:")
        print(f"Number of groups: {len(plot_data['groups'])}")
        for group in plot_data['groups']:
            print(f"Group '{group['title']}': {len(group['data'])} points")
            
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
