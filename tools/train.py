# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
from datetime import datetime

import mlflow
import mlflow.pytorch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='http://35.165.139.156:5000',
        help='MLflow tracking server URI')
    parser.add_argument(
        '--mlflow-experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (defaults to config filename)')
    parser.add_argument(
        '--mlflow-run-name',
        type=str,
        default=None,
        help='MLflow run name (defaults to timestamp)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Set experiment name (default to config filename without extension)
    if args.mlflow_experiment_name is None:
        experiment_name = osp.splitext(osp.basename(args.config))[0]
    else:
        experiment_name = args.mlflow_experiment_name
    
    # Set run name (default to timestamp)
    if args.mlflow_run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_name = args.mlflow_run_name
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Warning: Could not set up MLflow experiment: {e}")
        experiment_id = None

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Log configuration parameters
        try:
            # Log basic training parameters
            mlflow.log_param("config_file", args.config)
            mlflow.log_param("max_epochs", cfg.train_cfg.get('max_epochs', 'unknown'))
            mlflow.log_param("val_interval", cfg.train_cfg.get('val_interval', 'unknown'))
            mlflow.log_param("batch_size", cfg.train_dataloader.get('batch_size', 'unknown'))
            mlflow.log_param("learning_rate", cfg.optim_wrapper.optimizer.get('lr', 'unknown'))
            mlflow.log_param("weight_decay", cfg.optim_wrapper.optimizer.get('weight_decay', 'unknown'))
            
            # Log model architecture parameters
            if hasattr(cfg, 'model'):
                mlflow.log_param("model_type", cfg.model.get('type', 'unknown'))
                if 'backbone' in cfg.model:
                    mlflow.log_param("backbone_type", cfg.model.backbone.get('type', 'unknown'))
                if 'neck' in cfg.model:
                    mlflow.log_param("neck_type", cfg.model.neck.get('type', 'unknown'))
                if 'bbox_head' in cfg.model:
                    mlflow.log_param("bbox_head_type", cfg.model.bbox_head.get('type', 'unknown'))
                    mlflow.log_param("num_classes", cfg.model.bbox_head.get('num_classes', 'unknown'))
            
            # Log dataset information
            if hasattr(cfg, 'train_dataloader') and 'dataset' in cfg.train_dataloader:
                dataset_cfg = cfg.train_dataloader.dataset
                mlflow.log_param("dataset_type", dataset_cfg.get('type', 'unknown'))
                if 'metainfo' in dataset_cfg and 'classes' in dataset_cfg.metainfo:
                    mlflow.log_param("dataset_classes", str(dataset_cfg.metainfo.classes))
            
            # Log system information
            mlflow.log_param("launcher", cfg.launcher)
            mlflow.log_param("amp_enabled", args.amp)
            mlflow.log_param("auto_scale_lr", args.auto_scale_lr)
            
            print(f"MLflow run started: {run.info.run_id}")
            print(f"Experiment: {experiment_name}")
            print(f"Run name: {run_name}")
            
        except Exception as e:
            print(f"Warning: Could not log parameters to MLflow: {e}")

        # Add MLflow hook to the config BEFORE building the runner
        try:
            from mmdet.engine.hooks.mlflow_hook import MLflowHook
            mlflow_hook_config = dict(
                type='MLflowHook',
                tracking_uri=args.mlflow_tracking_uri,
                experiment_name=experiment_name,
                run_name=run_name,
                log_interval=1,  # Log every epoch instead of every N iterations
                log_artifacts=True,
                log_model=True
            )
            if 'custom_hooks' not in cfg:
                cfg.custom_hooks = []
            cfg.custom_hooks.append(mlflow_hook_config)
            print("MLflow hook added to training configuration")
        except ImportError:
            print("Warning: MLflowHook not found. Using basic MLflow logging.")
        except Exception as e:
            print(f"Warning: Could not add MLflow hook: {e}")

        # build the runner from config (AFTER adding hooks)
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        try:
            runner.train()
            
            # Log final model artifacts
            try:
                # Log the best checkpoint if it exists
                best_ckpt_path = osp.join(cfg.work_dir, 'best_coco_bbox_mAP_epoch_*.pth')
                import glob
                best_ckpts = glob.glob(best_ckpt_path)
                if best_ckpts:
                    best_ckpt = max(best_ckpts, key=os.path.getctime)
                    mlflow.log_artifact(best_ckpt, "checkpoints")
                    print(f"Logged best checkpoint: {best_ckpt}")
                
                # Log the latest checkpoint
                latest_ckpt_path = osp.join(cfg.work_dir, 'latest.pth')
                if osp.exists(latest_ckpt_path):
                    mlflow.log_artifact(latest_ckpt_path, "checkpoints")
                    print(f"Logged latest checkpoint: {latest_ckpt_path}")
                
                # Log training logs
                log_files = glob.glob(osp.join(cfg.work_dir, "*.log"))
                for log_file in log_files:
                    mlflow.log_artifact(log_file, "logs")
                
                # Log config file
                config_copy_path = osp.join(cfg.work_dir, osp.basename(args.config))
                if osp.exists(config_copy_path):
                    mlflow.log_artifact(config_copy_path, "configs")
                
                print("Training artifacts logged to MLflow")
                
            except Exception as e:
                print(f"Warning: Could not log artifacts to MLflow: {e}")
                
        except Exception as e:
            print(f"Training failed: {e}")
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
            raise
        else:
            mlflow.log_param("training_status", "completed")
            print("Training completed successfully")


if __name__ == '__main__':
    main()
