# MLflow Integration with MMDetection Training

This document describes the MLflow integration added to the MMDetection training pipeline.

## Overview

The `train.py` script has been enhanced with comprehensive MLflow tracking capabilities that automatically log:

- **Parameters**: Model configuration, training hyperparameters, dataset information
- **Metrics**: Training/validation losses, learning rates, memory usage, timing
- **Artifacts**: Model checkpoints, training logs, configuration files
- **Models**: Trained models for easy deployment and comparison

## Features

### Automatic Parameter Logging
- Model architecture details (backbone, neck, head types)
- Training configuration (epochs, batch size, learning rate, etc.)
- Dataset information (type, classes)
- System configuration (launcher, AMP, auto-scaling)

### Real-time Metric Tracking
- Training losses and metrics
- Validation performance metrics
- Learning rate schedules
- Memory usage and timing information
- GPU utilization

### Artifact Management
- Best and latest model checkpoints
- Training logs and configuration files
- Model artifacts for deployment

## Usage

### Basic Usage

```bash
# Use default MLflow settings
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py
```

### Custom MLflow Configuration

```bash
# Specify custom MLflow tracking server
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py \
    --mlflow-tracking-uri http://your-mlflow-server:5000

# Set custom experiment and run names
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py \
    --mlflow-experiment-name "rtmdet_finetune_experiments" \
    --mlflow-run-name "run_001_baseline"
```

### Command Line Arguments

- `--mlflow-tracking-uri`: MLflow tracking server URI (default: http://35.165.139.156:5000)
- `--mlflow-experiment-name`: Name of the MLflow experiment (default: config filename)
- `--mlflow-run-name`: Name of the MLflow run (default: timestamp)

## MLflow Hook

The integration includes a custom `MLflowHook` that provides:

- **Real-time logging**: Metrics logged during training iterations and epochs
- **Checkpoint tracking**: Automatic logging of model checkpoints
- **Artifact management**: Training logs, configs, and model files
- **Error handling**: Graceful fallback if MLflow is unavailable

### Hook Configuration

The hook is automatically added to the training configuration. You can also manually configure it in your config file:

```python
custom_hooks = [
    dict(
        type='MLflowHook',
        tracking_uri='http://35.165.139.156:5000',
        experiment_name='my_experiment',
        run_name='my_run',
        log_interval=50,
        log_artifacts=True,
        log_model=True
    )
]
```

## What Gets Logged

### Parameters
- `config_file`: Path to the configuration file
- `work_dir`: Working directory for training
- `max_epochs`: Maximum number of training epochs
- `val_interval`: Validation interval
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `weight_decay`: Weight decay parameter
- `model_type`: Type of detection model
- `backbone_type`: Backbone architecture
- `neck_type`: Neck architecture
- `bbox_head_type`: Detection head type
- `num_classes`: Number of detection classes
- `dataset_type`: Dataset type
- `dataset_classes`: List of class names

### Metrics
- `train/loss`: Training loss values
- `val/*`: Validation metrics (mAP, precision, recall, etc.)
- `learning_rate`: Current learning rate
- `memory/cuda_mb`: GPU memory usage
- `time/train_epoch_time`: Training time per epoch
- `time/val_epoch_time`: Validation time per epoch

### Artifacts
- `checkpoints/`: Model checkpoint files
- `logs/`: Training log files
- `configs/`: Configuration files
- `model/`: Trained model for deployment

## MLflow UI

Access the MLflow UI to view and compare your experiments:

```bash
# Start MLflow UI (if running locally)
mlflow ui

# Or access the remote server
# Navigate to http://35.165.139.156:5000 in your browser
```

## Error Handling

The integration includes comprehensive error handling:

- **Connection failures**: Training continues if MLflow server is unavailable
- **Logging failures**: Individual metric logging failures don't stop training
- **Artifact failures**: Missing files are handled gracefully
- **Warning messages**: Clear feedback about MLflow status

## Requirements

Make sure you have MLflow installed:

```bash
pip install mlflow
```

## Example Training Run

```bash
# Start training with MLflow tracking
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py \
    --mlflow-experiment-name "rtmdet_finetune" \
    --mlflow-run-name "baseline_experiment"

# The output will show:
# MLflow run started: abc123def456
# Experiment: rtmdet_finetune
# Run name: baseline_experiment
# MLflow hook added to training configuration
# Training artifacts logged to MLflow
```

## Benefits

1. **Experiment Tracking**: Compare different model configurations and hyperparameters
2. **Reproducibility**: All training parameters and artifacts are logged
3. **Model Management**: Easy access to trained models for deployment
4. **Performance Monitoring**: Real-time tracking of training progress
5. **Collaboration**: Share experiments with team members through MLflow UI
6. **Automation**: No manual logging required - everything is automatic

This integration makes MMDetection training fully compatible with modern MLOps workflows and experiment tracking best practices.
