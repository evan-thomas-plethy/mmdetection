"""Shared helpers for RTMDet bbox AP evaluation and prediction paths."""

import json
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

BBOX_STATS_NAMES = [
    'AP', 'AP .5', 'AP .75', 'AP (S)', 'AP (M)', 'AP (L)',
    'AR@100', 'AR@300', 'AR@1000', 'AR@100 (S)', 'AR@100 (M)', 'AR@100 (L)',
]


def load_coco_eval_modules():
    try:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    except ImportError:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    return COCO, COCOeval


def sanitize_path_tag(name):
    return str(name).replace('.', '_').replace(' ', '_')


def infer_dataset_name(ann_file, dataset_name=None):
    if dataset_name:
        return sanitize_path_tag(dataset_name)
    ann_path = Path(ann_file)
    if ann_path.parent.name == 'annotations':
        return sanitize_path_tag(ann_path.parent.parent.name)
    return sanitize_path_tag(ann_path.stem)


def predictions_output_path(
    backend,
    model_path,
    ann_file,
    predictions_dir=None,
    dataset_name=None,
    max_samples=None,
):
    backend = backend.lower()
    if backend not in ('coreml', 'tflite'):
        raise ValueError(f'backend must be coreml or tflite, got {backend!r}')

    model_tag = sanitize_path_tag(Path(model_path).stem)
    dataset_tag = infer_dataset_name(ann_file, dataset_name)
    filename = f'{backend}_{model_tag}_{dataset_tag}'
    if max_samples is not None:
        filename += f'_n{max_samples}'
    out_dir = Path(predictions_dir) if predictions_dir is not None else Path('predictions')
    return out_dir / f'{filename}.detections.json'


def infer_precision_from_path(path):
    stem = Path(path).stem.lower()
    if '_int8' in stem or stem.endswith('int8'):
        return 'int8'
    if any(tag in stem for tag in ('_fp16', '_float16', 'float_16', 'float16')):
        return 'fp16'
    if any(tag in stem for tag in ('_float32', '_fp32', 'float_32', 'float32')):
        return 'fp32'
    return 'fp32'


def column_labels_for_paths(paths):
    paths = [Path(p) for p in paths]
    precisions = [infer_precision_from_path(p) for p in paths]
    precision_counts = {}
    for prec in precisions:
        precision_counts[prec] = precision_counts.get(prec, 0) + 1

    labels = {}
    for path, prec in zip(paths, precisions):
        labels[path] = prec if precision_counts[prec] == 1 else path.stem
    return labels


def compute_bbox_ap_from_predictions(ann_file, predictions):
    """Run COCO bbox AP on a list of detection prediction dicts."""
    COCO, COCOeval = load_coco_eval_modules()
    coco = COCO(str(ann_file))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name

    coco_dt = coco.loadRes(pred_file)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    Path(pred_file).unlink(missing_ok=True)
    return {
        name: float(value)
        for name, value in zip(BBOX_STATS_NAMES, coco_eval.stats)
    }


def print_ap_comparison_table(baseline_metrics, variant_metrics_by_label, baseline_label='fp32'):
    labels = list(variant_metrics_by_label)
    header = f'{"Metric":<12} {baseline_label:>12}'
    for label in labels:
        header += f' {label:>12}'
    for label in labels:
        header += f' {"d" + label:>12}'
    print(header)
    print('-' * len(header))

    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        row = f'{metric:<12} {baseline_val:>12.4f}'
        for label in labels:
            row += f' {variant_metrics_by_label[label][metric]:>12.4f}'
        for label in labels:
            delta = variant_metrics_by_label[label][metric] - baseline_val
            row += f' {delta:>+12.4f}'
        print(row)

    print('-' * len(header))
    for label in labels:
        ap_delta = variant_metrics_by_label[label]['AP'] - baseline_metrics['AP']
        print(f'Primary bbox AP delta ({label} - {baseline_label}): {ap_delta:+.4f}')


def compare_two_predictions_to_report(
    ann_file,
    prediction_paths,
    report_path,
    export_script_hint,
):
    prediction_paths = [Path(p) for p in prediction_paths]
    if len(prediction_paths) != 2:
        raise ValueError('Exactly two --predictions paths are required')

    for path in prediction_paths:
        if not path.exists():
            raise FileNotFoundError(
                f'Predictions not found: {path}. Run {export_script_hint} first.')

    labels = column_labels_for_paths(prediction_paths)
    buffer = StringIO()
    with redirect_stdout(buffer):
        print(f'Annotation file: {ann_file}')
        for path in prediction_paths:
            print(f'{labels[path]} predictions: {path}')
        print()

        metrics = {}
        for path in prediction_paths:
            label = labels[path]
            print(f'Computing bbox AP for {label} predictions...')
            with open(path) as f:
                preds = json.load(f)
            metrics[label] = compute_bbox_ap_from_predictions(ann_file, preds)
            print()

        baseline_path = prediction_paths[0]
        baseline_label = labels[baseline_path]
        variant_label = labels[prediction_paths[1]]

        print('=' * 96)
        print('COCO bbox AP comparison (same metric as training save_best=coco/bbox_mAP)')
        print('=' * 96)
        print_ap_comparison_table(
            metrics[baseline_label],
            {variant_label: metrics[variant_label]},
            baseline_label=baseline_label,
        )

    report = buffer.getvalue()
    sys.stdout.write(report)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f'Wrote report to {report_path}', file=sys.stderr)
