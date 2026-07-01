"""Compare COCO bbox AP from two exported RTMDet detection prediction JSON files."""

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from det_eval_common import compare_two_predictions_to_report
from model_paths import ANN_FILE, AP_EXPORTED_REPORT


def main():
    parser = argparse.ArgumentParser(
        description='Compare COCO bbox AP from two exported detection JSONs.')
    parser.add_argument(
        '--predictions',
        nargs=2,
        type=Path,
        metavar='PATH',
        required=True,
        help='Two prediction JSON paths (first = baseline).',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ANN_FILE,
        help='COCO val annotations JSON.',
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=AP_EXPORTED_REPORT,
        help='Report output path.',
    )
    args = parser.parse_args()

    compare_two_predictions_to_report(
        args.ann_file,
        args.predictions,
        args.report,
        'tools/exported_export_val_predictions.py',
    )


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f'ERROR: Comparison failed - {exc}', file=sys.stderr)
        sys.exit(1)
