"""Export COCO val bbox predictions from exported RTMDet models (CoreML / TFLite)."""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from det_eval_common import predictions_output_path
from model_paths import PREDICTIONS_DIR
from rtmdet_exported_val_inference import export_val_predictions


def main():
    parser = argparse.ArgumentParser(
        description='Export COCO val bbox predictions from CoreML/TFLite RTMDet models.')
    parser.add_argument(
        '--models',
        nargs='+',
        type=Path,
        required=True,
        help='Exported model paths (.mlmodel or .tflite).',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        help='COCO val annotations JSON (person_keypoints JSON with bbox fields is OK).',
    )
    parser.add_argument(
        '--img-prefix',
        type=Path,
        required=True,
        help='Val images directory.',
    )
    parser.add_argument(
        '--dataset-name',
        default=None,
        help='Dataset tag for output filename (default: inferred from ann path).',
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=PREDICTIONS_DIR,
        help='Directory for prediction JSON output.',
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.05,
        help='Detection score threshold.',
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.6,
        help='NMS IoU threshold.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of val images.',
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Re-run inference even if prediction JSON exists.',
    )
    args = parser.parse_args()

    if not args.ann_file.exists():
        print(f'ERROR: Annotations not found: {args.ann_file}', file=sys.stderr)
        sys.exit(1)
    if not args.img_prefix.exists():
        print(f'ERROR: Image prefix not found: {args.img_prefix}', file=sys.stderr)
        sys.exit(1)

    for model_path in args.models:
        if not model_path.exists():
            print(f'ERROR: Model not found: {model_path}', file=sys.stderr)
            sys.exit(1)

        backend = 'coreml' if model_path.suffix.lower() == '.mlmodel' else 'tflite'
        output_path = predictions_output_path(
            backend,
            model_path,
            args.ann_file,
            predictions_dir=args.predictions_dir,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
        )
        print(f'Exporting {model_path.name} -> {output_path}')
        export_val_predictions(
            model_path,
            args.ann_file,
            args.img_prefix,
            output_path,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_samples=args.max_samples,
            force_rerun=args.force_rerun,
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: Export failed - {exc}', file=sys.stderr)
        sys.exit(1)
