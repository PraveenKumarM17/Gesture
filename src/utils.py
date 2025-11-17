"""
Utility functions for Hand Sign Recognition Project
Streamlined version with only essential functions
"""

import cv2
import numpy as np
import json
from pathlib import Path

# ==================== IMAGE PREPROCESSING ====================

def preprocess_image(image, target_size=(64, 64), apply_clahe=True):
    """
    Preprocess single image for model prediction
    
    Args:
        image: Input image (BGR or Grayscale)
        target_size: Target size for model (width, height)
        apply_clahe: Apply CLAHE enhancement for better contrast
    
    Returns:
        Preprocessed image ready for model (normalized, reshaped)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for adaptive contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    # Add dimensions for model: (1, height, width, 1)
    preprocessed = np.expand_dims(normalized, axis=-1)  # Add channel dimension
    preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
    
    return preprocessed


# ==================== LABEL MANAGEMENT ====================

def load_class_labels(labels_file='../model/class_labels.json'):
    """
    Load class labels from JSON file
    
    Args:
        labels_file: Path to class labels JSON file
    
    Returns:
        List of class labels
    """
    labels_path = Path(labels_file)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_file} not found!")
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    return labels


def save_class_labels(labels, output_file='../model/class_labels.json'):
    """
    Save class labels to JSON file
    
    Args:
        labels: List of class labels
        output_file: Path to save labels
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"✅ Class labels saved to {output_file}")


# ==================== BOUNDING BOX EXTRACTION ====================

def extract_hand_roi(frame, hand_landmarks, padding_percent=0.2):
    """
    Extract hand region of interest from frame using MediaPipe landmarks
    
    Args:
        frame: Input frame (BGR image)
        hand_landmarks: MediaPipe hand landmarks
        padding_percent: Padding around bounding box (0.0 to 1.0)
    
    Returns:
        roi: Extracted hand region
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    h, w = frame.shape[:2]
    
    # Get landmark coordinates
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Calculate bounding box
    x_min = int(min(x_coords) * w)
    x_max = int(max(x_coords) * w)
    y_min = int(min(y_coords) * h)
    y_max = int(max(y_coords) * h)
    
    # Add padding
    box_width = x_max - x_min
    box_height = y_max - y_min
    padding_x = int(box_width * padding_percent)
    padding_y = int(box_height * padding_percent)
    
    x_min = max(x_min - padding_x, 0)
    y_min = max(y_min - padding_y, 0)
    x_max = min(x_max + padding_x, w)
    y_max = min(y_max + padding_y, h)
    
    # Extract ROI
    roi = frame[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)
    
    return roi, bbox


# ==================== PERFORMANCE TRACKING ====================

class PerformanceTracker:
    """Track FPS and frame processing performance"""
    
    def __init__(self, window_size=30):
        self.frame_times = []
        self.window_size = window_size
        self.last_time = None
    
    def update(self):
        """Update with current timestamp"""
        import time
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def reset(self):
        """Reset tracker"""
        self.frame_times = []
        self.last_time = None