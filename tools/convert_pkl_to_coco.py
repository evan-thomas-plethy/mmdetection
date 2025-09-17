#!/usr/bin/env python3
"""
Script to convert RTMDet results from pickle format to COCO JSON format.
Performs NMS to get the single highest confidence bbox per image,
then updates the COCO annotations with predicted bounding boxes.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import argparse


def calculate_iou(box1, boxes):
    """
    Calculate IoU between box1 and multiple boxes.
    
    Args:
        box1: Single bounding box [4] in (x1, y1, x2, y2) format
        boxes: Array of bounding boxes [N, 4] in (x1, y1, x2, y2) format
    
    Returns:
        Array of IoU values [N]
    """
    x1_min = np.maximum(box1[0], boxes[:, 0])
    y1_min = np.maximum(box1[1], boxes[:, 1])
    x2_max = np.minimum(box1[2], boxes[:, 2])
    y2_max = np.minimum(box1[3], boxes[:, 3])
    
    # Calculate intersection area
    intersection = np.maximum(0, x2_max - x1_min) * np.maximum(0, y2_max - y1_min)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou


def apply_nms_and_keep_best(bboxes, scores, labels=None, iou_threshold=0.5, 
                           score_threshold=0.05):
    """
    Apply Non-Maximum Suppression and keep only the highest confidence detection.
    
    Args:
        bboxes: Array of bounding boxes [N, 4] in (x1, y1, x2, y2) format
        scores: Array of confidence scores [N]
        labels: Array of class labels [N] (optional)
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
    
    Returns:
        tuple: (filtered_bboxes, filtered_scores, filtered_labels)
    """
    if bboxes is None or len(bboxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy arrays
    bboxes = np.array(bboxes)
    scores = np.array(scores) if scores is not None else np.ones(len(bboxes))
    labels = np.array(labels) if labels is not None else np.zeros(len(bboxes))
    
    # Filter by score threshold first
    valid_mask = scores >= score_threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    if len(bboxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by confidence score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    bboxes = bboxes[sorted_indices]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]
    
    # Apply NMS - keep only the best detection
    keep_indices = []
    remaining_indices = list(range(len(bboxes)))
    
    while len(remaining_indices) > 0:
        # Keep the highest scoring box
        current_idx = remaining_indices[0]
        keep_indices.append(current_idx)
        
        if len(remaining_indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = bboxes[current_idx:current_idx+1]
        remaining_boxes = bboxes[remaining_indices[1:]]
        
        # Calculate IoU manually
        ious = calculate_iou(current_box[0], remaining_boxes)
        
        # Keep boxes with IoU < threshold
        keep_mask = ious < iou_threshold
        remaining_indices = [remaining_indices[i+1] for i, keep in enumerate(keep_mask) if keep]
    
    # Return only the best detection (first one after sorting)
    if len(keep_indices) > 0:
        best_idx = keep_indices[0]
        return bboxes[best_idx:best_idx+1], scores[best_idx:best_idx+1], labels[best_idx:best_idx+1]
    else:
        return np.array([]), np.array([]), np.array([])


def xyxy_to_xywh(bbox):
    """
    Convert bounding box from (x1, y1, x2, y2) to (x, y, width, height) format.
    
    Args:
        bbox: Bounding box [4] in (x1, y1, x2, y2) format
    
    Returns:
        Bounding box [4] in (x, y, width, height) format
    """
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def process_pkl_results(results_path, iou_threshold=0.5, score_threshold=0.05):
    """
    Process pickle results and extract the best detection per image after NMS.
    
    Args:
        results_path: Path to the pickle results file
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering
    
    Returns:
        dict: Mapping from image_id to best detection (bbox, score, label)
    """
    print(f"Loading results from: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} results")
    
    image_detections = {}
    
    for i, result in enumerate(results):
        try:
            # Extract image ID
            img_id = None
            if 'img_id' in result:
                img_id = int(result['img_id'])  # Convert to Python int
            elif 'image_id' in result:
                img_id = int(result['image_id'])  # Convert to Python int
            else:
                print(f"Warning: No image ID found in result {i}")
                continue
            
            # Extract bounding boxes from different possible locations
            bboxes = None
            scores = None
            labels = None
            
            # Check pred_instances first (predicted bboxes)
            if 'pred_instances' in result:
                pred_instances = result['pred_instances']
                if 'bboxes' in pred_instances:
                    bboxes = pred_instances['bboxes']
                if 'scores' in pred_instances:
                    scores = pred_instances['scores']
                if 'labels' in pred_instances:
                    labels = pred_instances['labels']
            
            # Check direct bbox fields (alternative format)
            elif 'bboxes' in result:
                bboxes = result['bboxes']
                if 'scores' in result:
                    scores = result['scores']
                if 'labels' in result:
                    labels = result['labels']
            
            if bboxes is not None and len(bboxes) > 0:
                # Apply NMS and keep only the best detection
                filtered_bboxes, filtered_scores, filtered_labels = apply_nms_and_keep_best(
                    bboxes, scores, labels, iou_threshold, score_threshold
                )
                
                if len(filtered_bboxes) > 0:
                    # Store the best detection
                    best_bbox = filtered_bboxes[0]
                    best_score = filtered_scores[0]
                    best_label = filtered_labels[0] if len(filtered_labels) > 0 else 0
                    
                    image_detections[img_id] = {
                        'bbox': best_bbox,
                        'score': best_score,
                        'label': best_label
                    }
                    print(f"Image {img_id}: Best detection with score {best_score:.3f}")
                else:
                    print(f"Image {img_id}: No detections remaining after NMS")
            else:
                print(f"Image {img_id}: No bounding boxes found")
                
        except Exception as e:
            print(f"Error processing result {i}: {e}")
            continue
    
    print(f"Processed {len(image_detections)} images with detections")
    return image_detections


def update_coco_annotations(coco_json_path, image_detections, output_path):
    """
    Update COCO JSON annotations with predicted bounding boxes.
    
    Args:
        coco_json_path: Path to the original COCO JSON file
        image_detections: Dict mapping image_id to best detection
        output_path: Path to save the updated COCO JSON
    """
    print(f"Loading COCO annotations from: {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Loaded COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Create a mapping from image_id to annotations
    image_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_annotations:
            image_to_annotations[img_id] = []
        image_to_annotations[img_id].append(ann)
    
    # Update annotations with predicted bounding boxes
    updated_count = 0
    for img_id, detections in image_detections.items():
        if img_id in image_to_annotations:
            # Get the best detection
            best_bbox = detections['bbox']
            best_score = detections['score']
            best_label = detections['label']
            
            # Convert bbox from (x1, y1, x2, y2) to (x, y, width, height)
            coco_bbox = xyxy_to_xywh(best_bbox)
            
            # Update all annotations for this image - ONLY change the bbox field
            for ann in image_to_annotations[img_id]:
                ann['bbox'] = [float(x) for x in coco_bbox]  # Convert to Python floats
                updated_count += 1
            
            print(f"Updated {len(image_to_annotations[img_id])} annotations for image {img_id}")
        else:
            print(f"Warning: Image {img_id} not found in COCO annotations")
    
    # Save the updated COCO JSON
    print(f"Saving updated COCO annotations to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Updated {updated_count} annotations total")
    print(f"Saved updated COCO JSON to: {output_path}")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='Convert RTMDet pickle results to COCO JSON format')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results.pkl file')
    parser.add_argument('--coco-json', type=str, 
                       default='data/coco/annotations/instances_val2017.json',
                       help='Path to original COCO JSON file')
    parser.add_argument('--output', type=str,
                       default='data/coco/annotations/pred_bboxes.json',
                       help='Path to save updated COCO JSON')
    parser.add_argument('--score-threshold', type=float, default=0.05,
                       help='Score threshold for filtering detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    # Process pickle results
    image_detections = process_pkl_results(
        args.results, 
        args.iou_threshold, 
        args.score_threshold
    )
    
    # Update COCO annotations
    update_coco_annotations(
        args.coco_json,
        image_detections,
        args.output
    )
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
