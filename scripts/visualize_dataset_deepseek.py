import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import argparse

# Category names for VisDrone dataset
CATEGORY_NAMES = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others'
}

# Colors for different categories (BGR format for OpenCV)
CATEGORY_COLORS = {
    1: (0, 255, 0),     # pedestrian - green
    2: (0, 200, 0),     # people - dark green
    3: (255, 255, 0),   # bicycle - cyan
    4: (255, 0, 0),     # car - red
    5: (255, 100, 0),   # van - orange
    6: (0, 0, 255),     # truck - blue
    7: (255, 0, 255),   # tricycle - magenta
    8: (200, 0, 200),   # awning-tricycle - purple
    9: (0, 255, 255),   # bus - yellow
    10: (0, 150, 255),  # motor - light orange
    11: (128, 128, 128) # others - gray
}

def parse_annotation_file(annotation_path):
    """
    Parse the VisDrone annotation file.
    
    Returns:
        dict: frame_index -> list of annotations for that frame
    """
    annotations_by_frame = {}
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 10:
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                bbox_left = float(parts[2])
                bbox_top = float(parts[3])
                bbox_width = float(parts[4])
                bbox_height = float(parts[5])
                score = float(parts[6])
                category = int(parts[7])
                truncation = int(parts[8])
                occlusion = int(parts[9])
                
                # Only include valid annotations (score=1 for groundtruth)
                if score == 1:
                    annotation = {
                        'target_id': target_id,
                        'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                        'score': score,
                        'category': category,
                        'truncation': truncation,
                        'occlusion': occlusion
                    }
                    
                    if frame_idx not in annotations_by_frame:
                        annotations_by_frame[frame_idx] = []
                    annotations_by_frame[frame_idx].append(annotation)
    
    return annotations_by_frame

def draw_annotations(image, annotations):
    """
    Draw bounding boxes and labels on the image.
    """
    img_copy = image.copy()
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        target_id = ann['target_id']
        category = ann['category']
        occlusion = ann['occlusion']
        truncation = ann['truncation']
        
        # Get color for this category
        color = CATEGORY_COLORS.get(category, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Create label text
        category_name = CATEGORY_NAMES.get(category, 'unknown')
        label = f"ID:{target_id} {category_name}"
        
        # Add occlusion/truncation markers
        if occlusion == 1:
            label += " [occ partial]"
        elif occlusion == 2:
            label += " [occ heavy]"
        if truncation == 1:
            label += " [trunc]"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x, y - label_h - 5), (x + label_w, y), color, -1)
        
        # Draw label text
        cv2.putText(img_copy, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy

def visualize_sequence(sequence_path, annotation_path, start_frame=1, end_frame=None, delay=100):
    """
    Visualize a sequence of images with their annotations.
    
    Args:
        sequence_path: Path to the sequence folder containing images
        annotation_path: Path to the annotation file
        start_frame: Starting frame index
        end_frame: Ending frame index (None for all frames)
        delay: Delay between frames in milliseconds (0 for manual, negative for no wait)
    """
    # Parse annotations
    print(f"Loading annotations from {annotation_path}...")
    annotations_by_frame = parse_annotation_file(annotation_path)
    print(f"Loaded annotations for {len(annotations_by_frame)} frames")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
    
    if not image_files:
        print(f"No images found in {sequence_path}")
        return
    
    # Determine frame range
    if end_frame is None:
        end_frame = len(image_files)
    
    print(f"Visualizing frames {start_frame} to {end_frame}")
    
    # Create window
    cv2.namedWindow('VisDrone MOT Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('VisDrone MOT Visualization', 1280, 720)
    
    for frame_num in range(start_frame, end_frame + 1):
        # Image filename (zero-padded to 7 digits as in VisDrone)
        img_filename = f"{frame_num:07d}.jpg"
        img_path = os.path.join(sequence_path, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        # Get annotations for this frame
        annotations = annotations_by_frame.get(frame_num, [])
        
        # Draw annotations
        img_annotated = draw_annotations(img, annotations)
        
        # Add frame information
        info_text = f"Frame: {frame_num} | Objects: {len(annotations)}"
        cv2.putText(img_annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('VisDrone MOT Visualization', img_annotated)
        
        # Handle key press
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            print("Exiting...")
            break
        elif key == ord(' '):  # Space to pause
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('s'):  # S to save screenshot
            save_path = f"screenshot_frame_{frame_num}.jpg"
            cv2.imwrite(save_path, img_annotated)
            print(f"Screenshot saved: {save_path}")
        elif key == ord('f'):  # F to toggle fullscreen
            cv2.setWindowProperty('VisDrone MOT Visualization', cv2.WND_PROP_FULLSCREEN, 
                                  cv2.WINDOW_FULLSCREEN)
    
    cv2.destroyAllWindows()

def visualize_single_image(image_path, annotation_path, frame_index=None):
    """
    Visualize a single image with its annotations.
    
    Args:
        image_path: Path to a single image
        annotation_path: Path to the annotation file
        frame_index: Frame index (if None, extract from filename)
    """
    # Parse annotations
    annotations_by_frame = parse_annotation_file(annotation_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Determine frame index
    if frame_index is None:
        # Try to extract from filename
        filename = os.path.basename(image_path)
        try:
            frame_index = int(os.path.splitext(filename)[0])
        except ValueError:
            print("Could not determine frame index from filename. Please provide frame_index parameter.")
            return
    
    # Get annotations for this frame
    annotations = annotations_by_frame.get(frame_index, [])
    
    # Draw annotations
    img_annotated = draw_annotations(img, annotations)
    
    # Add info
    info_text = f"Frame: {frame_index} | Objects: {len(annotations)}"
    cv2.putText(img_annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display
    cv2.imshow('VisDrone Visualization', img_annotated)
    print(f"Displaying frame {frame_index} with {len(annotations)} objects")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize VisDrone MOT dataset')
    parser.add_argument('--sequence', type=str, 
                       default=r"C:\Users\HSSL77\Downloads\VisDrone2019-MOT-train\sequences\uav0000013_01073_v",
                       help='Path to sequence folder')
    parser.add_argument('--annotation', type=str,
                       default=r"C:\Users\HSSL77\Downloads\VisDrone2019-MOT-train\annotations\uav0000013_01073_v.txt",
                       help='Path to annotation file')
    parser.add_argument('--single-image', type=str,
                       help='Path to a single image file (overrides sequence mode)')
    parser.add_argument('--start', type=int, default=1,
                       help='Starting frame index')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending frame index')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in milliseconds (0 for manual, negative for no wait)')
    
    args = parser.parse_args()
    
    # Check if single image mode
    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"Image not found: {args.single_image}")
            return
        visualize_single_image(args.single_image, args.annotation)
    else:
        # Sequence mode
        if not os.path.exists(args.sequence):
            print(f"Sequence folder not found: {args.sequence}")
            return
        if not os.path.exists(args.annotation):
            print(f"Annotation file not found: {args.annotation}")
            return
        
        visualize_sequence(args.sequence, args.annotation, args.start, args.end, args.delay)

if __name__ == "__main__":
    # Example usage without command line arguments
    # You can also run it directly with the default paths
    
    # For sequence visualization:
    sequence_path = r"C:\Users\HSSL77\Downloads\VisDrone2019-MOT-train\sequences\uav0000126_00001_v"
    annotation_path = r"C:\Users\HSSL77\Downloads\VisDrone2019-MOT-train\annotations\uav0000126_00001_v.txt"
    
    # Check if files exist
    if os.path.exists(sequence_path) and os.path.exists(annotation_path):
        print("Starting visualization...")
        print("Controls:")
        print("  q/ESC - Quit")
        print("  SPACE - Pause/Resume")
        print("  s - Save screenshot")
        print("  f - Toggle fullscreen")
        visualize_sequence(sequence_path, annotation_path, start_frame=1, delay=100)
    else:
        print("Default paths not found. Please provide correct paths.")
        # Fall back to command line argument parsing
        main()