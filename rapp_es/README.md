# Energy Saving RL Module (rapp_es)

**Self-contained Energy Saving Reinforcement Learning module for the Maveric platform.**

This module provides training and inference services for Energy Saving (ES) rApp. It is designed to integrate seamlessly with existing Kafka-based worker architectures of maveric_platform.

## What This Module Does

- **Training**: Trains RL agents to optimize cell tower energy consumption while maintaining coverage
- **Inference**: Predicts optimal cell configurations (ON/OFF states and tilt angles) for given time periods
- **Self-contained**: All dependencies included, no external imports needed
- **Async-ready**: Designed for Kafka worker integration

## Structure

```
rapp_es/
├── requirements.txt       # Main module requirements (includes dependencies)
├── rl_manager.py          # Main API - sync & async functions
├── rl_trainer.py          # Core training logic
├── rl_predictor.py        # Core inference logic
├── rl_energy_saving_env.py # RL environment
├── data_preprocessor.py   # Data preprocessing utilities
└── dependencies/          # Self-contained dependencies
    ├── requirements.txt   # Combined dependencies (apps + radp)
    ├── apps/
    │   └── requirements.txt # CCO engine dependencies
    └── radp/
        └── requirements.txt # Digital twin dependencies
```

## Quick Integration Guide

### Step 1: Install Dependencies

```bash
cd rapp_es
pip install -r requirements.txt
```

### Step 2: Import in Kafka Worker

```python
from rapp_es.rl_manager import train_es_rl_async, infer_es_rl_async
```

### Step 3: Add ES Support to Worker

```python
def _handle_message(self, message: str) -> None:
    # ... existing parsing logic ...

    if rapp_id == "ES":
        # Add ES support with async
        asyncio.run(self._handle_es_training(
            model_id, tenant_id, bdt_id, dataset_id, params, db
        ))
    else:
        # ... existing logic for other rapp_ids ...

async def _handle_es_training(self, model_id, tenant_id, bdt_id, dataset_id, params, db):
    """Handle ES-specific training requests"""
    try:
        self._update_model(tenant_id, model_id, status="training", db=db)

        # ES Training - this is the key call
        trained_model_path = await train_es_rl_async(
            train_days=params.get("train_days", [0, 1, 2, 3]),
            total_timesteps=params.get("total_timesteps", 48000),
            # ... add your S3 artifact paths here
        )

        # Mark as ready
        self._update_model(
            tenant_id, model_id,
            status="ready",
            artifacts=[f"s3://bucket/{tenant_id}/rapps/ES/{model_id}/model.zip"],
            db=db
        )

    except Exception as e:
        self._update_model(
            tenant_id, model_id,
            status="failed",
            metrics={"error": str(e)},
            db=db
        )
```

## API Functions

### Training

```python
await train_es_rl_async(
    train_days=[0, 1, 2, 3],        # Days to use for training
    total_timesteps=48000,          # RL training iterations
    bdt_model_path="path/to/bdt.pickle",
    topology_path="path/to/topology.csv",
    config_path="path/to/config.csv",
    # ... other optional paths
)
# Returns: path to trained model file
```

### Inference

```python
# Console output (default behavior)
await infer_es_rl_async(
    target_tick=12,                 # Hour of day (0-23)
    rl_model_path="path/to/model.zip",
    topology_path="path/to/topology.csv"
)
# Outputs: Optimal cell configurations to console/logs

# Return structured data for JSON export/storage
results = await infer_es_rl_async(
    target_tick=12,
    rl_model_path="path/to/model.zip",
    topology_path="path/to/topology.csv",
    return_results=True
)
# Returns: List of dicts with cell predictions
# Example: [{"cell_id": "cell_1_0", "predicted_state": "ON", "predicted_cell_el_deg": 10.0}]
```

## Kafka Message Format

The API should send messages in this format:

```json
{
  "spec": "maveric.rapp.train.v1",
  "tenant_id": "tenant-123",
  "rapp_id": "ES",
  "model_id": "model-456",
  "bdt_id": "bdt-789",
  "dataset_id": "dataset-101",
  "params": {
    "train_days": [0, 1, 2, 3],
    "total_timesteps": 48000
  }
}
```

## Required Data Files

Place these files in S3 bucket or local paths:

### For Training:

- `bdt_model_map.pickle` - Pre-trained Bayesian Digital Twin model
- `topology.csv` - Cell tower layout configuration
- `config.csv` - Initial cell configuration
- `generated_data/Day_*/ue_data_gym_ready/*.csv` - Preprocessed UE data

### For Inference:

- `energy_saver_agent.zip` - Trained RL model (output from training)
- `topology.csv` - Cell tower layout configuration

## Testing

```bash
# Test the module (runs sync functions by default)
cd rapp_es && python rl_manager.py

# Test async functions (edit rl_manager.py: try_async = True)
cd rapp_es && python rl_manager.py
```

## Key Benefits

- **Drop-in integration** - Minimal changes to existing worker
- **Async-ready** - Perfect for Kafka architecture
- **Self-contained** - No external dependency conflicts
- **Production-tested** - Uses proven RADP components
- **Simple API** - Just 2 main functions to call

## Troubleshooting

1. **Missing dependencies** - Run `pip install -r requirements.txt`
2. **Import errors** - Ensure `rapp_es/` is in Python path
3. **Training fails** - Check data file paths and formats
4. **Integration issues** - See example worker code above
