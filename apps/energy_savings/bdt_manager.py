# bdt_manager.py

# This module is intended to manage the training process of the Bayesian Digital Twin (BDT) model
# and to facilitate the downloading of the resulting model artifact.
import logging
import os
import subprocess
import sys

import pandas as pd

try:
    from radp.client.client import RADPClient
    from radp.client.helper import ModelStatus, RADPHelper
except ImportError:
    print("FATAL: Could not import RADP client modules. Ensure project root is in PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BDTManager:
    """
    The management of the training process for the Bayesian Digital Twin model,
    as well as the downloading of the resulting model, is performed by this class.
    """

    def __init__(self, topology_path: str, training_data_path: str, model_path: str):
        self.topology_path = topology_path
        self.training_data_path = training_data_path
        self.model_path = model_path
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train(self, model_id: str, container_name: str):
        # Every failure raises, so a pipeline (or notebook cell) stops here instead of
        # carrying on without a model.
        logger.info(f"Initiating BDT model training for model_id: {model_id}...")
        radp_client = RADPClient()
        radp_helper = RADPHelper(radp_client)

        logger.info(f"Loading topology from {self.topology_path}")
        topology_df = pd.read_csv(self.topology_path)
        logger.info(f"Loading training data from {self.training_data_path}")
        training_data_df = pd.read_csv(self.training_data_path)

        response = radp_client.train(
            model_id=model_id,
            params={},
            ue_training_data=training_data_df,
            topology=topology_df,
            model_update=False,
        )
        if isinstance(response, dict) and "error" in response:
            # A re-run with the same model_id is rejected by the API; reuse the model it already holds.
            if response.get("field") == "model_id" and "already exists" in response["error"]:
                logger.warning(f"Model '{model_id}' already exists in RADP; reusing it instead of retraining.")
            else:
                raise RuntimeError(f"RADP rejected the training request for '{model_id}': {response}")
        else:
            logger.info(f"Training request sent for model_id: {model_id}. Waiting for completion...")

        status: ModelStatus = radp_helper.resolve_model_status(
            model_id, wait_interval=30, max_attempts=120, verbose=True
        )
        if not status.success:
            raise RuntimeError(f"BDT model training failed: {status.error_message}")

        logger.info(f"BDT model '{model_id}' trained successfully in backend.")
        self._download_model_from_container(model_id, container_name)

    def _download_model_from_container(self, model_id: str, container_name: str):
        # The downloading of the trained model file from the specified Docker container
        # to the local file system is performed by this method.
        container_path = f"/srv/radp/models/{model_id}/model.pickle"
        logger.info(f"Attempting to download model from '{container_name}:{container_path}' to '{self.model_path}'...")
        command = ["docker", "cp", f"{container_name}:{container_path}", self.model_path]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError("'docker' command not found. Please ensure Docker is installed and in your PATH.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"'docker cp' from '{container_name}' failed: {e.stderr.strip()}. The container name must match "
                "the stack you started: 'radp_dev-training-1' (dc-dev.yml) or 'radp_prod-training-1' (dc-prod.yml)."
            ) from e
        logger.info("Model downloaded successfully from Docker container.")
