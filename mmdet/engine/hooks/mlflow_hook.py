# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Sequence, Union

import mlflow
import mlflow.pytorch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MLflowHook(Hook):
    """MLflow hook for logging metrics, parameters, and artifacts during training.

    This hook integrates MLflow tracking with MMDetection training pipeline,
    automatically logging training metrics, validation results, and model artifacts.

    Args:
        tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): Name of the MLflow experiment.
        run_name (str): Name of the MLflow run.
        log_interval (int): Interval for logging training metrics. Defaults to 50.
        log_artifacts (bool): Whether to log model artifacts. Defaults to True.
        log_model (bool): Whether to log the trained model. Defaults to True.
    """

    def __init__(self,
                 tracking_uri: str,
                 experiment_name: str,
                 run_name: str,
                 log_interval: int = 50,
                 log_artifacts: bool = True,
                 log_model: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_interval = log_interval
        self.log_artifacts = log_artifacts
        self.log_model = log_model
        self.current_run = None

    def before_run(self, runner: Runner) -> None:
        """Initialize MLflow connection before training starts."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Check if we're already in an MLflow run context
            try:
                self.current_run = mlflow.active_run()
                if self.current_run is not None:
                    runner.logger.info(f"Using existing MLflow run: {self.current_run.info.run_id}")
                else:
                    # Get or create experiment and start new run
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(self.experiment_name)
                    else:
                        experiment_id = experiment.experiment_id
                    
                    self.current_run = mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=self.run_name
                    )
                    runner.logger.info(f"MLflow run started: {self.current_run.info.run_id}")
            except Exception:
                # If no active run, try to start one
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                else:
                    experiment_id = experiment.experiment_id
                
                self.current_run = mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=self.run_name
                )
                runner.logger.info(f"MLflow run started: {self.current_run.info.run_id}")
            
            runner.logger.info(f"Experiment: {self.experiment_name}")
            runner.logger.info(f"Run name: {self.run_name}")
            
        except Exception as e:
            runner.logger.warning(f"Failed to initialize MLflow: {e}")
            self.current_run = None

    def after_train_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict = None,
                        outputs: Optional[dict] = None) -> None:
        """Store training metrics for epoch-level logging."""
        # We'll log training metrics at the epoch level instead of iteration level
        # This method is kept for compatibility but doesn't log to MLflow
        pass

    def after_train_epoch(self, runner: Runner) -> None:
        """Log training metrics after each epoch."""
        if self.current_run is None:
            return
            
        try:
            # Log epoch number
            mlflow.log_metric("epoch", runner.epoch, step=runner.epoch)
            
            # Log current learning rate
            if hasattr(runner, 'optim_wrapper') and hasattr(runner.optim_wrapper, 'optimizer'):
                lr = runner.optim_wrapper.optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", lr, step=runner.epoch)
            
            # Log training time
            if hasattr(runner, 'message_hub') and 'time' in runner.message_hub.runtime_info:
                time_info = runner.message_hub.runtime_info['time']
                if 'train_time' in time_info:
                    mlflow.log_metric("time/train_epoch_time", time_info['train_time'], step=runner.epoch)
            
            # Log memory usage
            if hasattr(runner, 'message_hub') and 'memory' in runner.message_hub.runtime_info:
                memory_info = runner.message_hub.runtime_info['memory']
                if 'cuda' in memory_info:
                    mlflow.log_metric("memory/cuda_mb", memory_info['cuda'], step=runner.epoch)
            
            # Log training loss from message hub if available
            if hasattr(runner, 'message_hub') and 'train' in runner.message_hub.runtime_info:
                train_info = runner.message_hub.runtime_info['train']
                if 'loss' in train_info:
                    loss_value = train_info['loss']
                    if isinstance(loss_value, (int, float)):
                        mlflow.log_metric("train/loss", loss_value, step=runner.epoch)
                    elif isinstance(loss_value, dict):
                        for key, value in loss_value.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"train/{key}", value, step=runner.epoch)
            
            runner.logger.info(f"Logged training metrics for epoch {runner.epoch}")
            
        except Exception as e:
            runner.logger.warning(f"Failed to log epoch metrics: {e}")

    def after_val_epoch(self, runner: Runner, metrics: Optional[dict] = None) -> None:
        """Log validation metrics after each validation epoch."""
        if self.current_run is None:
            return
            
        try:
            if metrics:
                # Log validation metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"val/{key}", value, step=runner.epoch)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                mlflow.log_metric(f"val/{key}_{sub_key}", sub_value, step=runner.epoch)
            
            # Log validation time
            if hasattr(runner, 'message_hub') and 'time' in runner.message_hub.runtime_info:
                time_info = runner.message_hub.runtime_info['time']
                if 'val_time' in time_info:
                    mlflow.log_metric("time/val_epoch_time", time_info['val_time'], step=runner.epoch)
            
            runner.logger.info(f"Logged validation metrics for epoch {runner.epoch}")
            
        except Exception as e:
            runner.logger.warning(f"Failed to log validation metrics: {e}")

    def after_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Log checkpoint artifacts after saving."""
        if self.current_run is None or not self.log_artifacts:
            return
            
        try:
            # Log checkpoint file
            checkpoint_path = checkpoint.get('filepath', '')
            if osp.exists(checkpoint_path):
                mlflow.log_artifact(checkpoint_path, "checkpoints")
                runner.logger.info(f"Logged checkpoint: {checkpoint_path}")
            
            # Log model if it's the best checkpoint
            if 'best_score' in checkpoint and self.log_model:
                try:
                    # Log the model using MLflow's PyTorch integration
                    model = runner.model
                    if hasattr(model, 'module'):
                        model = model.module
                    
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=f"{self.experiment_name}_{self.run_name}"
                    )
                    runner.logger.info("Logged model to MLflow")
                except Exception as e:
                    runner.logger.warning(f"Failed to log model: {e}")
            
        except Exception as e:
            runner.logger.warning(f"Failed to log checkpoint artifacts: {e}")

    def after_run(self, runner: Runner) -> None:
        """Clean up MLflow artifacts after training completes."""
        try:
            if self.current_run and self.log_artifacts:
                # Log final artifacts
                work_dir = runner.work_dir
                
                # Log training logs
                log_files = []
                for root, dirs, files in os.walk(work_dir):
                    for file in files:
                        if file.endswith('.log'):
                            log_files.append(osp.join(root, file))
                
                for log_file in log_files:
                    mlflow.log_artifact(log_file, "logs")
                
                # Log config file
                config_files = []
                for root, dirs, files in os.walk(work_dir):
                    for file in files:
                        if file.endswith('.py') and 'config' in file.lower():
                            config_files.append(osp.join(root, file))
                
                for config_file in config_files:
                    mlflow.log_artifact(config_file, "configs")
                
                runner.logger.info("Logged training artifacts to MLflow")
            
            # Note: We don't end the MLflow run here since it's managed by the outer context
            runner.logger.info("MLflow hook cleanup completed")
                
        except Exception as e:
            runner.logger.warning(f"Failed to finalize MLflow artifacts: {e}")
