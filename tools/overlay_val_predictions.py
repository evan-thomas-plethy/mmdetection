"""Render validation overlay MP4 from exported RTMDet detection predictions JSON."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from det_eval_common import infer_precision_from_path
from model_paths import OVERLAYS_DIR

DEFAULT_FPS = 10
PRECISION_COLORS_BGR = {
    'fp32': (0, 255, 0),
    'fp16': (0, 200, 255),
    'int8': (255, 0, 255),
}


def overlay_output_video_path(predictions_path, overlays_dir=None):
    predictions_path = Path(predictions_path)
    overlays_dir = Path(overlays_dir or OVERLAYS_DIR)
    filename = predictions_path.name
    if filename.endswith('.json'):
        filename = f'{filename[:-5]}.mp4'
    return overlays_dir / filename


def _load_coco(ann_file):
    try:
        from xtcocotools.coco import COCO
    except ImportError:
        from pycocotools.coco import COCO
    return COCO(str(ann_file))


def _load_predictions_by_image(predictions_path):
    with open(predictions_path) as f:
        predictions = json.load(f)
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred['image_id']].append(pred)
    for image_id in by_image:
        by_image[image_id].sort(key=lambda item: item.get('score', 0.0), reverse=True)
    return by_image


def _draw_bbox_xywh(image, bbox_xywh, color, thickness=2, label=None):
    x, y, w, h = bbox_xywh
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def render_overlay_video(
    predictions_path,
    ann_file,
    images_dir,
    output_path,
    max_images=None,
    draw_gt=True,
):
    coco = _load_coco(ann_file)
    predictions_by_image = _load_predictions_by_image(predictions_path)
    color = PRECISION_COLORS_BGR.get(
        infer_precision_from_path(predictions_path), (0, 255, 0))

    img_ids = coco.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    target_size = None

    for index, image_id in enumerate(img_ids, start=1):
        img_info = coco.loadImgs(image_id)[0]
        image_path = images_dir / img_info['file_name']
        if not image_path.exists():
            print(f'Warning: missing image {image_path}')
            continue

        canvas = cv2.imread(str(image_path))
        if canvas is None:
            print(f'Warning: could not read {image_path}')
            continue

        if draw_gt:
            ann_ids = coco.getAnnIds(imgIds=image_id)
            for ann in coco.loadAnns(ann_ids):
                if 'bbox' in ann:
                    _draw_bbox_xywh(canvas, ann['bbox'], (128, 128, 128), thickness=1)

        for pred in predictions_by_image.get(image_id, [])[:1]:
            score = pred.get('score', 0.0)
            label = f'{score:.2f}'
            _draw_bbox_xywh(canvas, pred['bbox'], color, thickness=2, label=label)

        if target_size is None:
            height, width = canvas.shape[:2]
            target_width = ((width + 15) // 16) * 16
            target_height = ((height + 15) // 16) * 16
            target_size = (target_width, target_height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, DEFAULT_FPS, target_size)
            if not writer.isOpened():
                raise RuntimeError(f'Could not open video writer: {output_path}')

        if canvas.shape[1] != target_size[0] or canvas.shape[0] != target_size[1]:
            canvas = cv2.resize(canvas, target_size)
        writer.write(canvas)

        if index % 100 == 0:
            print(f'Rendered {index}/{len(img_ids)} frames')

    if writer is not None:
        writer.release()
        print(f'Wrote overlay video: {output_path}')
    else:
        raise RuntimeError('No frames rendered; check images-dir and predictions JSON')


def main():
    parser = argparse.ArgumentParser(
        description='Render val overlay MP4 from exported RTMDet detections JSON.')
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Detection predictions JSON from exported_export_val_predictions.py.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        required=True,
        help='COCO val annotations JSON.',
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Val images directory.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output MP4 path (default: overlays/<predictions>.mp4).',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Limit number of frames (smoke test).',
    )
    parser.add_argument(
        '--no-gt',
        action='store_true',
        help='Do not draw ground-truth boxes.',
    )
    args = parser.parse_args()

    output_path = args.output or overlay_output_video_path(args.predictions)
    render_overlay_video(
        args.predictions,
        args.ann_file,
        args.images_dir,
        output_path,
        max_images=args.max_images,
        draw_gt=not args.no_gt,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: Overlay failed - {exc}', file=sys.stderr)
        sys.exit(1)
