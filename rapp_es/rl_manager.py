# rl_manager.py
# Energy Saving RL functions extracted from main_app.py for modular usage

import asyncio
import logging
import os
import sys
import time
from typing import List, Optional

# --- Python Path Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # /path/to/maveric/apps/energy_savings
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR)) # /path/to/maveric
sys.path.insert(0, PROJECT_ROOT)

# --- Local Module Imports ---
try:
    from rl_predictor import run_rl_prediction
    from rl_trainer import run_rl_training
except ImportError as e:
    print(f"FATAL: Could not import a required local module: {e}")
    print("Please ensure all required modules are available in the project.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def train_es_rl(
    train_days: List[int] = [0, 1, 2, 3],
    total_timesteps: int = 48000,
    bdt_model_path: Optional[str] = None,
    base_ue_data_dir: Optional[str] = None,
    topology_path: Optional[str] = None,
    config_path: Optional[str] = None,
    rl_model_path: Optional[str] = None,
    log_dir: Optional[str] = None
) -> str:
    """
    Train the Energy Saving Reinforcement Learning agent.

    Required Files:
        - bdt_model_map.pickle: Pre-trained BDT model file
        - data/topology.csv: Cell tower layout configuration
        - data/config.csv: Initial cell tower configuration
        - generated_data/Day_*/ue_data_gym_ready/: Preprocessed UE data for each training day
          (Note: Use preprocess_es_data() function first if you only have raw ue_data_per_tick/ files)

    Args:
        train_days: List of day numbers to use for RL training (default: [0, 1, 2, 3])
        total_timesteps: Total timesteps for RL training (default: 48000)
        bdt_model_path: Path to the BDT model pickle file (default: APP_DIR/bdt_model_map.pickle)
        base_ue_data_dir: Base directory containing UE data (default: APP_DIR/generated_data)
        topology_path: Path to the topology CSV file (default: APP_DIR/data/topology.csv)
        config_path: Path to the config CSV file (default: APP_DIR/data/config.csv)
        rl_model_path: Output path for the trained RL model (default: APP_DIR/energy_saver_agent.zip)
        log_dir: Directory for RL training logs (default: APP_DIR/rl_training_logs)

    Returns:
        str: Path to the trained RL model file

    Raises:
        FileNotFoundError: If required input files are not found
        Exception: If training fails
    """
    # Set default paths if not provided
    if bdt_model_path is None:
        bdt_model_path = os.path.join(APP_DIR, "bdt_model_map.pickle")
    if base_ue_data_dir is None:
        base_ue_data_dir = os.path.join(APP_DIR, "generated_data")
    if topology_path is None:
        topology_path = os.path.join(APP_DIR, "data", "topology.csv")
    if config_path is None:
        config_path = os.path.join(APP_DIR, "data", "config.csv")
    if rl_model_path is None:
        rl_model_path = os.path.join(APP_DIR, "energy_saver_agent.zip")
    if log_dir is None:
        log_dir = os.path.join(APP_DIR, "rl_training_logs")

    # Validate required files exist
    required_files = [bdt_model_path, topology_path, config_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Validate base UE data directory exists
    if not os.path.exists(base_ue_data_dir):
        raise FileNotFoundError(f"Base UE data directory not found: {base_ue_data_dir}")

    logger.info(f"--- Starting RL Training on Days: {train_days} ---")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"BDT model path: {bdt_model_path}")
    logger.info(f"Output model path: {rl_model_path}")

    try:
        run_rl_training(
            bdt_model_path=bdt_model_path,
            base_ue_data_dir=base_ue_data_dir,
            training_days=train_days,
            topology_path=topology_path,
            config_path=config_path,
            rl_model_path=rl_model_path,
            log_dir=log_dir,
            total_timesteps=total_timesteps,
        )

        # Verify the model was actually created
        if not os.path.exists(rl_model_path):
            raise FileNotFoundError(f"Training failed: Model file not created at {rl_model_path}")

        logger.info("--- RL Training Completed Successfully ---")
        return rl_model_path

    except Exception as e:
        logger.error(f"RL Training failed: {e}")
        raise


def infer_es_rl(
    target_tick: int,
    rl_model_path: Optional[str] = None,
    topology_path: Optional[str] = None
) -> None:
    """
    Run inference with the trained Energy Saving RL agent for a specific tick.

    Required Files:
        - energy_saver_agent.zip: Trained RL model file
        - data/topology.csv: Cell tower layout configuration

    Args:
        target_tick: The specific tick/hour (0-23) for inference
        rl_model_path: Path to the trained RL model zip file (default: APP_DIR/energy_saver_agent.zip)
        topology_path: Path to the topology CSV file (default: APP_DIR/data/topology.csv)

    Raises:
        ValueError: If target_tick is not in valid range (0-23)
        FileNotFoundError: If required input files are not found
        Exception: If inference fails
    """
    # Validate target_tick
    if not (0 <= target_tick <= 23):
        raise ValueError(f"target_tick must be between 0-23, got: {target_tick}")

    # Set default paths if not provided
    if rl_model_path is None:
        rl_model_path = os.path.join(APP_DIR, "energy_saver_agent.zip")
    if topology_path is None:
        topology_path = os.path.join(APP_DIR, "data", "topology.csv")

    # Validate required files exist
    required_files = [rl_model_path, topology_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    logger.info(f"--- Starting Inference for Tick {target_tick} ---")
    logger.info(f"RL model path: {rl_model_path}")
    logger.info(f"Topology path: {topology_path}")

    try:
        run_rl_prediction(
            model_load_path=rl_model_path,
            topology_path=topology_path,
            target_tick=target_tick
        )
        logger.info("--- Inference Completed Successfully ---")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


# =============================================================================
# ASYNC WRAPPERS FOR KAFKA WORKER INTEGRATION
# =============================================================================

async def train_es_rl_async(
    train_days: List[int] = [0, 1, 2, 3],
    total_timesteps: int = 48000,
    bdt_model_path: Optional[str] = None,
    base_ue_data_dir: Optional[str] = None,
    topology_path: Optional[str] = None,
    config_path: Optional[str] = None,
    rl_model_path: Optional[str] = None,
    log_dir: Optional[str] = None
) -> str:
    """
    Async wrapper for ES RL training. Returns path to trained model.

    This is designed to be called from Kafka worker's _handle_message method.

    Args:
        Same as train_es_rl()

    Returns:
        str: Path to the trained RL model file

    Raises:
        Exception: If training fails
    """
    logger.info(f"Starting async ES RL training with {total_timesteps} timesteps...")

    # Run the synchronous training function in a thread pool
    loop = asyncio.get_event_loop()
    trained_model_path = await loop.run_in_executor(
        None,
        train_es_rl,
        train_days,
        total_timesteps,
        bdt_model_path,
        base_ue_data_dir,
        topology_path,
        config_path,
        rl_model_path,
        log_dir
    )

    logger.info(f"Async ES RL training completed: {trained_model_path}")
    return trained_model_path


async def infer_es_rl_async(
    target_tick: int,
    rl_model_path: Optional[str] = None,
    topology_path: Optional[str] = None
) -> None:
    """
    Async wrapper for ES RL inference.

    This is designed to be called from inference endpoints.

    Args:
        Same as infer_es_rl()

    Raises:
        ValueError: If target_tick is not in valid range (0-23)
        Exception: If inference fails
    """
    if not (0 <= target_tick <= 23):
        raise ValueError(f"target_tick must be between 0-23, got: {target_tick}")

    logger.info(f"Starting async ES RL inference for tick {target_tick}...")

    # Run the synchronous inference function in a thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        infer_es_rl,
        target_tick,
        rl_model_path,
        topology_path
    )

    logger.info(f"Async ES RL inference completed for tick {target_tick}")


if __name__ == "__main__":
    """
    Test sync and async functions.
    Note: Run data_preprocessor.py first to preprocess raw UE data if needed.
    """
    try_async = False  # Set to True to test async functions instead

    logger.info("=== Testing Energy Saving RL Functions ===")

    try:
        test_timesteps = 5000  # Reduced timesteps for quick test

        if try_async:
            # Test async functions
            async def async_test_flow():
                logger.info("=== Testing Async ES RL Functions ===")

                # Test async training
                logger.info(f"Testing train_es_rl_async with {test_timesteps} timesteps...")
                trained_model_path = await train_es_rl_async(
                    train_days=[0, 1],
                    total_timesteps=test_timesteps
                )
                logger.info(f"Async training completed: {trained_model_path}")

                # Sleep between operations
                logger.info("Sleeping for 2 seconds...")
                await asyncio.sleep(2)

                # Test async inference
                logger.info("Testing infer_es_rl_async...")
                await infer_es_rl_async(target_tick=12)
                logger.info("Async inference completed successfully.")

                logger.info("=== All async tests completed successfully ===")

            # Run async test flow
            asyncio.run(async_test_flow())

        else:
            # Test sync functions (default flow)
            logger.info("=== Testing Sync ES RL Functions ===")

            logger.info(f"Testing sync train_es_rl function with {test_timesteps} timesteps...")
            trained_model_path = train_es_rl(
                train_days=[0, 1, 2, 3],
                total_timesteps=test_timesteps
            )
            logger.info(f"Sync training completed. Model saved at: {trained_model_path}")

            # Sleep between operations
            logger.info("Sleeping for 2 seconds...")
            time.sleep(2)

            # Test sync inference function
            logger.info("Testing sync infer_es_rl function...")
            infer_es_rl(target_tick=3)
            logger.info("Sync inference completed successfully.")

            logger.info("=== All sync tests completed successfully ===")

        logger.info("=== All tests completed successfully ===")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)