"""
Hand Sign Recognition - IMPROVED Real-Time Detection with FAST TRACKING
Fixed preprocessing and better prediction smoothing
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from pathlib import Path
import sys
import json
import time


# FAST MODE variables
prev_bbox = None
bbox_history = deque(maxlen=5)
hand_lost_counter = 0
HAND_LOST_GRACE = 10
DETECTION_INTERVAL = 8


# ==================== CONFIGURATION ====================
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'hand_sign_model.h5'
LABELS_PATH = Path(__file__).parent.parent / 'model' / 'class_labels.json'

# ADJUSTED SETTINGS for better accuracy
CONFIDENCE_THRESHOLD = 0.40  # Lowered from 0.60
BUFFER_SIZE = 7  # Reduced from 10 for faster response
MIN_AGREEMENT = 0.50  # Increased from 0.40 for more stability

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# UI Colors (BGR)
COLOR_PRIMARY = (0, 255, 100)
COLOR_SECONDARY = (255, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (40, 40, 40)

# ==================== PREPROCESSING FUNCTION ====================

def preprocess_for_model(image, apply_clahe=False):
    """
    Preprocess image EXACTLY like training:
    grayscale → resize → normalize → reshape
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # (CLAHE is disabled now)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Resize to 64x64
    resized = cv2.resize(gray, (64, 64))

    # Normalize
    normalized = resized / 255.0

    # Add channel + batch dimension
    final = np.expand_dims(normalized, axis=-1)  # (64,64,1)
    final = np.expand_dims(final, axis=0)        # (1,64,64,1)

    return final


# ==================== LOAD MODEL & LABELS ====================

print("\n" + "=" * 80)
print("🚀 HAND SIGN RECOGNITION - IMPROVED REAL-TIME DETECTION")
print("=" * 80)

# Load model
if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please train the model first using train_model_fixed.py")
    exit(1)

print(f"📦 Loading model...")
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Load labels
if not LABELS_PATH.exists():
    print(f"❌ Labels file not found at {LABELS_PATH}")
    exit(1)

with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)

print(f"📋 Loaded {len(labels)} class labels")
print(f"   Classes: {', '.join(labels[:15])}..." if len(labels) > 15 else f"   Classes: {', '.join(labels)}")

# ==================== INITIALIZE MEDIAPIPE ====================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe initialized")

# ==================== INITIALIZE CAMERA ====================

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("📹 Camera initialized")

# ==================== INITIALIZE TRACKING ====================

pred_buffer = deque(maxlen=BUFFER_SIZE)
confidence_buffer = deque(maxlen=BUFFER_SIZE)

# Statistics
frame_count = 0
detection_count = 0
last_stable_sign = None
stable_count = 0

# Performance tracking
fps_times = deque(maxlen=30)
last_time = time.time()

print("\n" + "=" * 80)
print("🎮 CONTROLS:")
print("  Q - Quit")
print("  C - Clear prediction buffer")
print("  R - Reset statistics")
print("  S - Save screenshot with prediction")
print("  D - Toggle debug mode")
print("=" * 80)
print("\n💡 TIPS FOR BETTER ACCURACY:")
print("  • Ensure good, even lighting")
print("  • Keep hand centered in frame")
print("  • Hold sign steady for 1-2 seconds")
print("  • Use simple background")
print("  • Keep consistent distance from camera")
print("\n▶️  Starting detection...\n")

debug_mode = False

# ==================== UTILITY FUNCTIONS ====================

def calculate_fps():
    """Calculate FPS"""
    global last_time, fps_times
    current_time = time.time()
    fps_times.append(1.0 / (current_time - last_time) if current_time > last_time else 0)
    last_time = current_time
    return sum(fps_times) / len(fps_times) if fps_times else 0

def draw_info_panel(img, title, value, x, y, width, height):
    """Draw information panel"""
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_PANEL, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 2)
    cv2.putText(img, title, (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(img, str(value), (x + 10, y + 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_TEXT, 2)

def draw_confidence_bar(img, x, y, width, height, value):
    """Draw confidence bar"""
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill_width = int(value * width)
    
    if value > 0.8:
        color = COLOR_PRIMARY
    elif value > 0.6:
        color = COLOR_SECONDARY
    else:
        color = COLOR_WARNING
    
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 2)

# ==================== MAIN LOOP ====================

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame")
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1
    fps = calculate_fps()
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Decide whether to run full Mediapipe detection or use tracking
    run_full_detection = (frame_count % DETECTION_INTERVAL == 0) or (prev_bbox is None)
    
    # Variables for this frame
    current_sign = None
    current_confidence = 0.0
    hand_detected = False
    roi_for_debug = None
    
    # ---------------- FAST HAND TRACKING ----------------
    
    hand_detected = False
    curr_bbox = None
    
    # --- 1. Full detection (every DETECTION_INTERVAL frames) ---
    if run_full_detection:
        result = hands.process(rgb)
    
        if result and result.multi_hand_landmarks:
            hand_detected = True
            hand_lost_counter = 0
            detection_count += 1
    
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
    
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
    
                box_width = x_max - x_min
                box_height = y_max - y_min
    
                padding_x = int(box_width * 0.12)
                padding_y = int(box_height * 0.12)
    
                x_min = max(x_min - padding_x, 0)
                y_min = max(y_min - padding_y, 0)
                x_max = min(x_max + padding_x, w)
                y_max = min(y_max + padding_y, h)
    
                curr_bbox = [x_min, y_min, x_max, y_max]
                prev_bbox = curr_bbox
                bbox_history.append(curr_bbox)
    
    # --- 2. No detection → use previous bounding box (tracking mode) ---
    else:
        if prev_bbox is not None:
            hand_lost_counter += 1
            if hand_lost_counter <= HAND_LOST_GRACE:
                curr_bbox = prev_bbox
                hand_detected = True
    
    # --- 3. Smooth bbox and predict ---
    if curr_bbox:
        avg = np.mean(bbox_history, axis=0).astype(int)
        x_min, y_min, x_max, y_max = avg
    
        # Draw smoothed bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_PRIMARY, 3)
    
        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        roi_for_debug = roi.copy() if roi.size > 0 else None
    
        if roi.size > 0:
            try:
                preprocessed = preprocess_for_model(roi, apply_clahe=False)
                prediction = model.predict(preprocessed, verbose=0)
    
                current_confidence = float(np.max(prediction))
                predicted_idx = np.argmax(prediction)
                current_sign = labels[predicted_idx]
    
                pred_buffer.append(current_sign)
                confidence_buffer.append(current_confidence)
    
            except Exception as e:
                if debug_mode:
                    print("ROI/Prediction Error:", e)
    else:
        roi_for_debug = None
    
    # ==================== PREDICTION SMOOTHING ====================
    
    final_sign = None
    final_confidence = 0.0
    
    if pred_buffer and confidence_buffer:
        # Count occurrences
        counter = Counter(pred_buffer)
        most_common = counter.most_common(1)[0]
        sign = most_common[0]
        count = most_common[1]
        
        # Calculate agreement ratio
        agreement = count / len(pred_buffer)
        
        # Average confidence for this sign
        sign_confidences = [conf for s, conf in zip(pred_buffer, confidence_buffer) if s == sign]
        avg_confidence = sum(sign_confidences) / len(sign_confidences) if sign_confidences else 0
        
        # Only accept if meets thresholds
        if agreement >= MIN_AGREEMENT and avg_confidence >= CONFIDENCE_THRESHOLD:
            final_sign = sign
            final_confidence = avg_confidence
            
            # Track stability
            if final_sign == last_stable_sign:
                stable_count += 1
            else:
                stable_count = 1
                last_stable_sign = final_sign
        else:
            stable_count = 0
    
    # ==================== UI RENDERING ====================
    
    # Top panel
    panel_height = 140
    cv2.rectangle(frame, (0, 0), (w, panel_height), COLOR_PANEL, -1)
    cv2.line(frame, (0, panel_height), (w, panel_height), COLOR_PRIMARY, 3)
    
    # Main prediction display
    if final_sign and final_confidence >= CONFIDENCE_THRESHOLD:
        # Large sign display
        sign_text = f"Sign: {final_sign}"
        cv2.putText(frame, sign_text, (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, COLOR_TEXT, 3)
        
        # Confidence bar
        bar_x = 30
        bar_y = 85
        bar_width = 400
        bar_height = 25
        draw_confidence_bar(frame, bar_x, bar_y, bar_width, bar_height, final_confidence)
        
        # Confidence percentage
        conf_text = f"{final_confidence * 100:.1f}%"
        cv2.putText(frame, conf_text, (bar_x + bar_width + 15, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        # Stability indicator
        if stable_count > 10:
            status_text = "STABLE ✓"
            status_color = COLOR_PRIMARY
        else:
            status_text = "DETECTING..."
            status_color = COLOR_SECONDARY
        
        cv2.putText(frame, status_text, (w - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Buffer agreement
        if pred_buffer:
            counter = Counter(pred_buffer)
            most_common = counter.most_common(1)[0]
            agreement = (most_common[1] / len(pred_buffer)) * 100
            cv2.putText(frame, f"Agreement: {agreement:.0f}%", (w - 250, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    else:
        # No detection message
        if hand_detected and current_sign:
            msg = f"Low confidence ({current_confidence:.0%})"
            color = COLOR_WARNING
        elif hand_detected:
            msg = "Processing..."
            color = COLOR_SECONDARY
        else:
            msg = "Show hand sign"
            color = COLOR_WARNING
        
        cv2.putText(frame, msg, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # Info panels (bottom right)
    panel_width = 150
    panel_height = 70
    panel_spacing = 10
    start_x = w - panel_width - 20
    start_y = h - (panel_height + panel_spacing) * 3 - 20
    
    # FPS panel
    draw_info_panel(frame, "FPS", f"{fps:.1f}",
                   start_x, start_y, panel_width, panel_height)
    
    # Frame count panel
    draw_info_panel(frame, "FRAMES", frame_count,
                   start_x, start_y + panel_height + panel_spacing,
                   panel_width, panel_height)
    
    # Detection rate panel
    detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
    draw_info_panel(frame, "DETECTION", f"{detection_rate:.0f}%",
                   start_x, start_y + (panel_height + panel_spacing) * 2,
                   panel_width, panel_height)
    
    # Debug: Show ROI if available
    if debug_mode and roi_for_debug is not None:
        try:
            roi_display = cv2.resize(roi_for_debug, (200, 200))
            roi_display = cv2.cvtColor(roi_display, cv2.COLOR_BGR2GRAY) if len(roi_display.shape) == 3 else roi_display
            roi_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
            frame[10:210, w-220:w-20] = roi_display
            cv2.rectangle(frame, (w-220, 10), (w-20, 210), COLOR_PRIMARY, 2)
            cv2.putText(frame, "ROI", (w-210, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        except:
            pass
    
    # Bottom control bar
    control_text = "Q:Quit | C:Clear | R:Reset | S:Screenshot | D:Debug"
    text_size = cv2.getTextSize(control_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, control_text, (text_x, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Display frame
    cv2.imshow("Hand Sign Recognition - Improved", frame)
    
    # ==================== KEYBOARD CONTROLS ====================
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    
    elif key == ord('c'):
        pred_buffer.clear()
        confidence_buffer.clear()
        stable_count = 0
        last_stable_sign = None
        print("✓ Buffer cleared")
    
    elif key == ord('r'):
        frame_count = 0
        detection_count = 0
        stable_count = 0
        last_stable_sign = None
        pred_buffer.clear()
        confidence_buffer.clear()
        fps_times.clear()
        print("✓ Statistics reset")
    
    elif key == ord('s'):
        screenshot_path = Path(__file__).parent.parent / f"screenshot_{final_sign if final_sign else 'none'}_{frame_count}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        print(f"📸 Screenshot saved: {screenshot_path}")
    
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 80)
print("📊 SESSION STATISTICS")
print("=" * 80)
print(f"Total frames processed: {frame_count}")
print(f"Hands detected: {detection_count}")
print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
print(f"Average FPS: {fps:.1f}")
print("=" * 80)
print("✅ Application closed successfully")