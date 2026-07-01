"""RTMDet-nano exported-model val inference (CoreML / TFLite) -> COCO detections JSON."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

RTMDET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
RTMDET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
PERSON_CATEGORY_ID = 1


def _resolve_coreml_outputs(output):
    boxes = scores = None
    for value in output.values():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            scores = arr
    if boxes is None or scores is None:
        keys = list(output.keys())
        raise ValueError(f'Could not infer bbox/score outputs from CoreML keys: {keys}')
    return boxes, scores


def _resolve_tflite_outputs(interpreter):
    output_details = interpreter.get_output_details()
    boxes_index = scores_index = None
    for detail in output_details:
        name = detail.get('name', '')
        if name == 'Identity':
            boxes_index = detail['index']
        elif name == 'Identity_1':
            scores_index = detail['index']
    if boxes_index is None or scores_index is None:
        if len(output_details) < 2:
            raise ValueError('TFLite model must expose bbox and score outputs')
        boxes_index = output_details[0]['index']
        scores_index = output_details[1]['index']
    return boxes_index, scores_index


def _resolve_tflite_input_index(interpreter):
    for detail in interpreter.get_input_details():
        if detail.get('name') in ('batch_inputs.1', 'batch_inputs_1', 'input'):
            return detail['index']
    return interpreter.get_input_details()[0]['index']


def preprocess_image(image_path, input_shape=(320, 320)):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Image not found: {image_path}')

    original_height, original_width = image.shape[:2]
    target_height, target_width = input_shape

    scale = min(target_width / original_width, target_height / original_height)
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)
    dx = (target_width - scaled_width) / 2.0
    dy = (target_height - scaled_height) / 2.0

    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded[int(dy):int(dy) + scaled_height, int(dx):int(dx) + scaled_width] = resized

    scale_x = original_width / scaled_width if scaled_width else 1.0
    scale_y = original_height / scaled_height if scaled_height else 1.0
    meta = {
        'original_size': (original_height, original_width),
        'scale_factors': (scale_x, scale_y),
        'offsets': (dx, dy),
    }
    return padded, meta


def preprocess_for_coreml(image_path):
    padded, meta = preprocess_image(image_path)
    from PIL import Image

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), meta


def preprocess_for_tflite(image_path):
    padded, meta = preprocess_image(image_path)
    arr = padded.astype(np.float32)
    arr = (arr - RTMDET_MEAN) / RTMDET_STD
    return np.expand_dims(arr, axis=0), meta


def postprocess_detections(
    boxes,
    scores,
    meta,
    score_threshold=0.05,
    nms_threshold=0.6,
):
    if len(boxes.shape) == 3:
        boxes = boxes[0]
    if len(scores.shape) == 3:
        scores = scores[0]
    if len(scores.shape) == 2 and scores.shape[1] == 1:
        scores = scores.flatten()

    valid = scores >= score_threshold
    boxes = boxes[valid]
    scores = scores[valid]
    if len(boxes) == 0:
        return []

    nms_boxes = []
    for x1, y1, x2, y2 in boxes:
        nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        scores.tolist(),
        score_threshold,
        nms_threshold,
    )
    if len(indices) == 0:
        return []

    scale_x, scale_y = meta['scale_factors']
    dx, dy = meta['offsets']
    original_height, original_width = meta['original_size']

    detections = []
    flat_indices = indices.flatten().tolist() if hasattr(indices, 'flatten') else indices
    for idx in flat_indices:
        x, y, width, height = nms_boxes[int(idx)]
        score = float(scores[int(idx)])

        x = (x - dx) * scale_x
        y = (y - dy) * scale_y
        width *= scale_x
        height *= scale_y

        x = max(0.0, min(x, original_width - 1))
        y = max(0.0, min(y, original_height - 1))
        width = max(1.0, min(width, original_width - x))
        height = max(1.0, min(height, original_height - y))
        detections.append([x, y, width, height, score])

    detections.sort(key=lambda item: item[4], reverse=True)
    return detections


def load_model(model_path):
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    if suffix == '.mlmodel':
        import coremltools as ct

        return ct.models.MLModel(str(model_path)), 'coreml'
    if suffix == '.tflite':
        try:
            import ai_edge_litert.interpreter as litert
            interpreter = litert.Interpreter(model_path=str(model_path))
        except ImportError:
            import tensorflow.lite as tflite
            interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter, 'tflite'
    raise ValueError(f'Unsupported model format: {suffix}. Use .mlmodel or .tflite')


def run_inference(model, model_type, image_path, score_threshold=0.05, nms_threshold=0.6):
    if model_type == 'coreml':
        pil_image, meta = preprocess_for_coreml(image_path)
        output = model.predict({'batch_inputs_1': pil_image})
        boxes, scores = _resolve_coreml_outputs(output)
    else:
        input_data, meta = preprocess_for_tflite(image_path)
        input_index = _resolve_tflite_input_index(model)
        boxes_index, scores_index = _resolve_tflite_outputs(model)
        model.set_tensor(input_index, input_data)
        model.invoke()
        boxes = model.get_tensor(boxes_index)
        scores = model.get_tensor(scores_index)

    return postprocess_detections(
        np.asarray(boxes),
        np.asarray(scores),
        meta,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )


def load_coco_index(ann_file):
    with open(ann_file) as f:
        coco = json.load(f)

    images_by_file = {}
    for image in coco['images']:
        images_by_file[image['file_name']] = image['id']
    return images_by_file, coco.get('images', [])


def export_val_predictions(
    model_path,
    ann_file,
    img_prefix,
    output_path,
    score_threshold=0.05,
    nms_threshold=0.6,
    max_samples=None,
    force_rerun=False,
    keep_top_k=1,
):
    output_path = Path(output_path)
    if output_path.exists() and not force_rerun:
        print(f'Skipping existing predictions: {output_path}')
        return output_path

    ann_file = Path(ann_file)
    img_prefix = Path(img_prefix)
    images_by_file, images = load_coco_index(ann_file)
    if max_samples is not None:
        images = images[:max_samples]

    model, model_type = load_model(model_path)
    predictions = []

    for index, image_info in enumerate(images, start=1):
        file_name = image_info['file_name']
        image_path = img_prefix / file_name
        if not image_path.exists():
            print(f'Warning: missing image {image_path}')
            continue

        detections = run_inference(
            model,
            model_type,
            image_path,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        image_id = images_by_file.get(file_name, image_info['id'])
        for det in detections[:keep_top_k]:
            x, y, width, height, score = det
            predictions.append({
                'image_id': image_id,
                'category_id': PERSON_CATEGORY_ID,
                'bbox': [float(x), float(y), float(width), float(height)],
                'score': float(score),
            })

        if index % 100 == 0:
            print(f'Processed {index}/{len(images)} images')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f'Wrote {len(predictions)} detections to {output_path}')
    return output_path
