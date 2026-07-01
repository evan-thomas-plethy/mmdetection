"""Evaluate on a secondary validation dataset during detection training."""

import copy

import torch
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class GeneralValHook(Hook):
    """Evaluate on a general validation dataset to monitor forgetting.

    Runs after each primary validation epoch and logs metrics with a
    ``general_val/`` prefix (via CocoMetric ``prefix``).

    Args:
        dataloader (dict): Dataloader config for the general val dataset.
        evaluator (dict): Evaluator config (typically CocoMetric for bbox).
        interval (int): Run every N epochs. Default: 1.
        priority (int): Hook priority. Use below EMAHook (e.g. 48) so
            ``after_val_epoch`` runs before EMA restores training weights.
    """

    def __init__(
        self,
        dataloader: dict,
        evaluator: dict,
        interval: int = 1,
        priority: int = 48,
    ):
        self.priority = priority
        self.dataloader_cfg = copy.deepcopy(dataloader)
        self.evaluator_cfg = copy.deepcopy(evaluator)
        self.interval = interval
        self._dataloader = None
        self._evaluator = None

    def before_run(self, runner: Runner) -> None:
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg)
        dataloader_cfg.setdefault('persistent_workers', True)
        dataloader_cfg.setdefault('drop_last', False)
        dataloader_cfg.setdefault(
            'sampler', dict(type='DefaultSampler', shuffle=False))

        self._dataloader = Runner.build_dataloader(dataloader_cfg)

        evaluator_cfg = copy.deepcopy(self.evaluator_cfg)
        evaluator_cfg.setdefault('prefix', 'general_val')
        self._evaluator = Evaluator(evaluator_cfg)

        dataset_meta = getattr(self._dataloader.dataset, 'metainfo', None)
        if dataset_meta:
            for metric in self._evaluator.metrics:
                metric.dataset_meta = dataset_meta

        runner.logger.info(
            'GeneralValHook: Built dataloader with '
            f'{len(self._dataloader.dataset)} samples')

    def after_val_epoch(self, runner: Runner, metrics: dict = None) -> None:
        epoch = runner.epoch

        if epoch % self.interval != 0:
            return

        if self._dataloader is None or self._evaluator is None:
            runner.logger.warning(
                'GeneralValHook: dataloader or evaluator not initialized!')
            return

        runner.logger.info(f'Running general validation at epoch {epoch}...')

        model = runner.model
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                for data_batch in self._dataloader:
                    outputs = model.val_step(data_batch)
                    self._evaluator.process(
                        data_samples=outputs,
                        data_batch=data_batch,
                    )

            general_metrics = self._evaluator.evaluate(
                len(self._dataloader.dataset))

            runner.logger.info(
                f'General validation results: {general_metrics}')

            if runner.visualizer is not None:
                for key, value in general_metrics.items():
                    if isinstance(value, (int, float)):
                        runner.visualizer.add_scalar(key, value, step=epoch)

            runner.message_hub.update_info(
                'general_val_metrics', general_metrics)

        except Exception as e:
            runner.logger.error(f'GeneralValHook failed: {e}')
            import traceback
            runner.logger.error(traceback.format_exc())

        finally:
            if was_training:
                model.train()
