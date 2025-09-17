#!/usr/bin/env python3
"""
Script to load RTMDet results from results.pkl, overlay bounding boxes on images using OpenCV,
and save annotated images to the visualization folder.
"""

import pickle
import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path

def clear_output_directory(output_dir):
    """Clear all files in the output directory before processing."""
    if output_dir.exists():
        print(f"Clearing output directory: {output_dir}")
        # Remove all files and subdirectories
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"✓ Output directory cleared")
    else:
        print(f"Output directory does not exist, will be created: {output_dir}")

def draw_bboxes(image, bboxes, scores=None, labels=None, class_names=None, 
                score_threshold=0.05, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on an image."""
    if bboxes is None or len(bboxes) == 0:
        return image
    
    # Convert to numpy array if needed
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    
    # Ensure bboxes is 2D array
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape(1, -1)
    
    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Check score threshold if scores are provided
            if scores is not None and len(scores) > i:
                if scores[i] < score_threshold:
                    continue
            
            # Draw bounding box rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = ""
            if labels is not None and len(labels) > i:
                if class_names is not None and len(class_names) > labels[i]:
                    label_text = class_names[labels[i]]
                else:
                    label_text = f"Class {labels[i]}"
            
            if scores is not None and len(scores) > i:
                if label_text:
                    label_text += f": {scores[i]:.3f}"
                else:
                    label_text = f"{scores[i]:.3f}"
            
            # Draw label background
            if label_text:
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(image, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

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
        
        # Calculate IoU manually (simpler than using MMDetection's function)
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

def overlay_bboxes_on_images(results_path=None, images_dir=None, output_dir=None, 
                           score_threshold=0.05, class_names=None, iou_threshold=0.5):
    """Load RTMDet results and overlay bounding boxes on images."""
    
    # Get the paths - allow custom paths or use defaults
    script_dir = Path(__file__).parent
    if results_path is None:
        results_path = script_dir.parent / "results.pkl"  # Default to current directory
    else:
        results_path = Path(results_path)
    
    if images_dir is None:
        images_dir = script_dir.parent / "../data" / "coco" / "val2017"
    else:
        images_dir = Path(images_dir)
    
    if output_dir is None:
        output_dir = script_dir.parent / "visualization"
    else:
        output_dir = Path(output_dir)
    
    print(f"Looking for results file at: {results_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Clear output directory before processing
    clear_output_directory(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Error: File not found at {results_path}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    # Default class names for RTMDet (person detection)
    if class_names is None:
        class_names = ['person']  # Based on your config: num_classes=1
    
    try:
        # Load the pickle file
        print(f"Loading results from: {results_path}")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded {len(results)} results")
        
        # Process each result
        for i, result in enumerate(results):
            try:
                # Extract image path
                img_path = None
                if 'img_path' in result:
                    img_path = result['img_path']
                elif 'img_id' in result:
                    # Try to construct path from img_id
                    img_id = result['img_id']
                    img_path = f"{img_id:012d}.jpg"  # COCO format
                else:
                    print(f"Warning: No image path found in result {i}")
                    continue
                
                # Clean up image path
                if img_path.startswith('data/coco/val2017/'):
                    img_path = img_path[18:]  # Remove 'data/coco/val2017/' prefix
                elif img_path.startswith('val2017/'):
                    img_path = img_path[8:]   # Remove 'val2017/' prefix
                
                full_img_path = images_dir / img_path
                
                if not full_img_path.exists():
                    print(f"Warning: Image not found: {full_img_path}")
                    continue
                
                # Load image
                image = cv2.imread(str(full_img_path))
                if image is None:
                    print(f"Warning: Could not load image: {full_img_path}")
                    continue
                
                print(f"Processing image {i+1}/{len(results)}: {img_path}")
                
                # Extract bounding boxes from different possible locations
                bboxes = None
                scores = None
                labels = None
                
                # Check pred_instances first (predicted bboxes)
                if 'pred_instances' in result:
                    pred_instances = result['pred_instances']
                    if 'bboxes' in pred_instances:
                        bboxes = pred_instances['bboxes']
                        print(f"  Found predicted bboxes: {len(bboxes)} detections")
                    
                    if 'scores' in pred_instances:
                        scores = pred_instances['scores']
                        print(f"  Found predicted scores: {len(scores)} scores")
                    
                    if 'labels' in pred_instances:
                        labels = pred_instances['labels']
                        print(f"  Found predicted labels: {len(labels)} labels")
                
                # Check gt_instances (ground truth bboxes) if no predictions
                elif 'gt_instances' in result:
                    gt_instances = result['gt_instances']
                    if 'bboxes' in gt_instances:
                        bboxes = gt_instances['bboxes']
                        print(f"  Using ground truth bboxes: {len(bboxes)} detections")
                    
                    if 'labels' in gt_instances:
                        labels = gt_instances['labels']
                        print(f"  Using ground truth labels: {len(labels)} labels")
                
                # Check direct bbox fields (alternative format)
                elif 'bboxes' in result:
                    bboxes = result['bboxes']
                    if 'scores' in result:
                        scores = result['scores']
                    if 'labels' in result:
                        labels = result['labels']
                    print(f"  Found direct bbox fields: {len(bboxes)} detections")
                
                if bboxes is not None and len(bboxes) > 0:
                    # Apply NMS and keep only the best detection
                    print(f"  Applying NMS (IoU threshold: {iou_threshold})")
                    print(f"  Before NMS: {len(bboxes)} detections")
                    
                    filtered_bboxes, filtered_scores, filtered_labels = apply_nms_and_keep_best(
                        bboxes, scores, labels, iou_threshold, score_threshold
                    )
                    
                    print(f"  After NMS: {len(filtered_bboxes)} detections remaining")
                    
                    if len(filtered_bboxes) > 0:
                        # Draw only the best bounding box
                        image_with_bboxes = draw_bboxes(
                            image.copy(), filtered_bboxes, filtered_scores, filtered_labels, 
                            class_names, score_threshold
                        )
                        
                        # Generate output filename
                        img_name = Path(img_path).stem
                        output_filename = f"{img_name}_bboxes.jpg"
                        output_path = output_dir / output_filename
                        
                        # Save annotated image
                        cv2.imwrite(str(output_path), image_with_bboxes)
                        print(f"  Saved annotated image: {output_filename}")
                    else:
                        print(f"  No detections remaining after NMS")
                        # Save original image without annotations
                        img_name = Path(img_path).stem
                        output_filename = f"{img_name}_bboxes.jpg"
                        output_path = output_dir / output_filename
                        cv2.imwrite(str(output_path), image)
                        print(f"  Saved original image: {output_filename}")
                else:
                    print(f"  No bounding boxes found in result")
                
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcessing complete! Annotated images saved to: {output_dir}")
        
        # Create video from all annotated images
        print("\nCreating video from annotated images...")
        create_video_from_images(output_dir)
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

def create_video_from_images(images_dir, fps=10, output_path=None):
    """Create a video from all images in the directory."""
    if output_path is None:
        output_path = images_dir.parent / "result.mp4"
    
    # Get all image files (look for bbox images instead of keypoint images)
    image_files = sorted([f for f in images_dir.glob("*_bboxes.jpg")])
    
    if not image_files:
        print("No annotated images found to create video")
        return
    
    print(f"Found {len(image_files)} images to create video")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print("Could not read first image")
        return
    
    # Find the maximum dimensions across all images to ensure compatibility
    max_width, max_height = 0, 0
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        if img is not None:
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)
    
    # Use a standard resolution that can accommodate all images
    # Round up to nearest multiple of 16 for better codec compatibility
    target_width = ((max_width + 15) // 16) * 16
    target_height = ((max_height + 15) // 16) * 16
    
    print(f"Maximum image dimensions: {max_width}x{max_height}")
    print(f"Target video dimensions: {target_width}x{target_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    if not video_writer.isOpened():
        print("Could not open video writer")
        return
    
    # Add each image to video
    for i, image_file in enumerate(image_files):
        print(f"Adding frame {i+1}/{len(image_files)}: {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is not None:
            # Resize image to target dimensions
            resized_image = cv2.resize(image, (target_width, target_height))
            video_writer.write(resized_image)
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")

def main():
    """Main function with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RTMDet bounding box results')
    parser.add_argument('--results', type=str, help='Path to results.pkl file')
    parser.add_argument('--images', type=str, help='Path to images directory')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--score-threshold', type=float, default=0.05, 
                       help='Score threshold for displaying detections')
    parser.add_argument('--class-names', nargs='+', default=['person'],
                       help='Class names for labels')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    overlay_bboxes_on_images(
        results_path=args.results,
        images_dir=args.images,
        output_dir=args.output,
        score_threshold=args.score_threshold,
        class_names=args.class_names,
        iou_threshold=args.iou_threshold
    )

if __name__ == "__main__":
    main()
