# MLflow Integration with MMDetection Training

This document describes the MLflow integration for the MMDetection training pipeline using MMEngine's built-in `MLflowVisBackend`.

## Overview

MLflow integration is achieved through MMEngine's visualization backend system. When enabled, **all training and validation metrics are automatically logged** to MLflow without any custom hooks or code modifications.

## Quick Start

### Enable MLflow via Command Line

```bash
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py \
    --mlflow-tracking-uri http://35.165.139.156:5000
```

That's it! All metrics will be logged automatically.

### With Custom Experiment/Run Names

```bash
python tools/train.py configs/rtmdet/rtmdet_nano_320_8xb32_coco_finetune.py \
    --mlflow-tracking-uri http://35.165.139.156:5000 \
    --mlflow-experiment-name "rtmdet_finetune" \
    --mlflow-run-name "baseline_v1"
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow-tracking-uri` | MLflow server URI. **Required to enable logging.** | http://35.165.139.156:5000 |
| `--mlflow-experiment-name` | Experiment name | Config filename |
| `--mlflow-run-name` | Run name | Timestamp |

## Alternative: Configure via Config File

You can also enable MLflow directly in your config file:

```python
# Add to your config file
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='http://35.165.139.156:5000',
        exp_name='my_experiment',
        run_name='my_run',
    )
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
```

## What Gets Logged

### Automatically Logged Metrics (Every Epoch)

**Training metrics:**
- `loss` - Total training loss
- `loss_cls` - Classification loss
- `loss_bbox` - Bounding box loss
- `lr` - Learning rate
- `time` - Iteration time
- `data_time` - Data loading time
- `memory` - GPU memory usage

**Validation metrics:**
- `coco/bbox_mAP` - COCO mAP
- `coco/bbox_mAP_50` - mAP at IoU=0.5
- `coco/bbox_mAP_75` - mAP at IoU=0.75
- And all other COCO metrics

### Automatically Logged Artifacts (End of Training)

- Configuration file (`.py`)
- Training logs (`.log`)
- JSON files (`.json`)
- YAML files (`.yaml`)

### Automatically Logged Config Parameters

The full config is flattened and logged as parameters, including:
- Model architecture
- Training hyperparameters
- Dataset settings
- Optimizer configuration

## How It Works

1. **LoggerHook** collects all training/validation metrics
2. **Visualizer** receives the metrics via `add_scalars()`
3. **MLflowVisBackend** forwards them to MLflow via `mlflow.log_metrics()`
4. On training completion, `close()` logs all artifacts automatically

This is MMEngine's standard logging pipeline - no custom code needed!

## MLflow UI

View your experiments at your MLflow server:

```bash
# If running locally
mlflow ui

# Or access remote server
# http://35.165.139.156:5000
```

## Requirements

```bash
pip install mlflow
```

## Comparison with Custom Hook Approach

| Feature | MLflowVisBackend | Custom Hook |
|---------|------------------|-------------|
| Code changes | None (config only) | Custom hook code |
| Metric access | Automatic via LoggerHook | Manual extraction |
| Artifact logging | Automatic | Manual |
| Config logging | Automatic (flattened) | Manual |
| Maintenance | Built into MMEngine | Custom maintenance |

The `MLflowVisBackend` approach is recommended as it's:
- **Simpler** - Just add to config or use CLI flag
- **More reliable** - Uses MMEngine's logging pipeline
- **Feature-complete** - Logs metrics, config, and artifacts automatically
- **Maintained** - Part of MMEngine, not custom code

## Example Output

```
MLflow logging enabled:
  Tracking URI: http://35.165.139.156:5000
  Experiment: rtmdet_nano_320_8xb32_coco_finetune
  Run name: 20251216_235500
```

All metrics will appear in the MLflow UI with proper step numbers corresponding to epochs.
