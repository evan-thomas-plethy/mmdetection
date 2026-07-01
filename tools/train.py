# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from datetime import datetime

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
    # MLflow arguments
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='http://52.41.68.196:5000',
        help='MLflow tracking server URI. If provided, enables MLflow logging.')
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


def add_mlflow_backend(cfg, args):
    """Add MLflowVisBackend to the visualizer if MLflow is enabled."""
    if args.mlflow_tracking_uri is None:
        return cfg
    
    # Set experiment name (default to config filename without extension)
    if args.mlflow_experiment_name is None:
        exp_name = osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = args.mlflow_experiment_name
    
    # Set run name (default to timestamp)
    if args.mlflow_run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_name = args.mlflow_run_name
    
    # Create MLflowVisBackend config
    mlflow_backend = dict(
        type='MLflowVisBackend',
        tracking_uri=args.mlflow_tracking_uri,
        exp_name=exp_name,
        run_name=run_name,
    )
    
    # Ensure vis_backends exists and add MLflow backend
    if not hasattr(cfg, 'vis_backends') or cfg.vis_backends is None:
        cfg.vis_backends = [dict(type='LocalVisBackend')]
    
    # Check if MLflowVisBackend is already configured
    has_mlflow = any(
        backend.get('type') == 'MLflowVisBackend' 
        for backend in cfg.vis_backends
    )
    
    if not has_mlflow:
        cfg.vis_backends.append(mlflow_backend)
        print(f"MLflow logging enabled:")
        print(f"  Tracking URI: {args.mlflow_tracking_uri}")
        print(f"  Experiment: {exp_name}")
        print(f"  Run name: {run_name}")
    
    # Update visualizer to use the new vis_backends
    if hasattr(cfg, 'visualizer') and cfg.visualizer is not None:
        cfg.visualizer.vis_backends = cfg.vis_backends
    
    return cfg


def main():
    args = parse_args()

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

    # Add MLflow backend if tracking URI is provided
    cfg = add_mlflow_backend(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
