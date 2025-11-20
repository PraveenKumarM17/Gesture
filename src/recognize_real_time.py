import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from pathlib import Path
import json
import time
from googletrans import Translator

# ==================== CONFIGURATION ====================
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'hand_sign_model.h5'
LABELS_PATH = Path(__file__).parent.parent / 'model' / 'class_labels.json'
SENTENCE_FILE = Path(__file__).parent.parent / 'sentence.txt'

CONFIDENCE_THRESHOLD = 0.45
BUFFER_SIZE = 10
MIN_AGREEMENT = 0.5
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Colors
COLOR_PRIMARY = (0, 255, 0)       # Green for detected hand
COLOR_SECONDARY = (255, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (40, 40, 40)

# ==================== DICTIONARY FOR KANNADA ====================
KAN_DICT = {
    # ----------------- Animals -----------------
    'cat': 'ಬೆಕ್ಕು',
    'dog': 'ನಾಯಿ',
    'cow': 'ಹಸು',
    'elephant': 'ಆನೆ',
    'lion': 'ಸಿಂಹ',
    'tiger': 'ಹುಲಿ',
    'horse': 'ಕುದುರೆ',
    'bear': 'ಕರಡಿ',
    'sheep': 'ಕುರಿ',
    'goat': 'ಮೇಕೆ',
    'rabbit': 'ಮೊಲ',
    'monkey': 'ಕೋತಿ',
    'frog': 'ಕಪ್ಪೆ',
    'snake': 'ಹಾವು',
    'parrot': 'ಗಿಳಿ',
    'crow': 'ಕಾಗೆ',
    'peacock': 'ನವಿಲು',
    'owl': 'ಗೂಬೆ',
    'duck': 'ಬಾತುಕೋಳಿ',
    'fish': 'ಮೀನು',
    'whale': 'ಥಿಮಿಂಗಿಲ',
    # ----------------- Greetings -----------------
    'hello': 'ನಮಸ್ಕಾರ',
    'hi': 'ಹಾಯ್',
    'good morning': 'ಶುಭೋದಯ',
    'good afternoon': 'ಶುಭ ಮಧ್ಯಾಹ್ನ',
    'good evening': 'ಶುಭ ಸಂಜೆ',
    'good night': 'ಶುಭ ರಾತ್ರಿ',
    'how are you': 'ನೀವು ಹೇಗಿದ್ದೀರಾ',
    'i am fine': 'ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ',
    'thank you': 'ಧನ್ಯವಾದಗಳು',
    'thanks': 'ಧನ್ಯವಾದ',
    'please': 'ದಯವಿಟ್ಟು',
    'sorry': 'ಕ್ಷಮಿಸಿ',
    'welcome': 'ಸ್ವಾಗತ',
    'bye': 'ವಿದಾಯ',
    'see you': 'ಮತ್ತೆ ಭೇಟಿಯಾಗೋಣ',
    'have a nice day': 'ಒಳ್ಳೆಯ ದಿನವಾಗಲಿ',
    'congratulations': 'ಹೃದಯಪೂರ್ವಕ ಅಭಿನಂದನೆಗಳು',
    'happy birthday': 'ಹುಟ್ಟುಹಬ್ಬದ ಶುಭಾಶಯಗಳು',
    'good luck': 'ಶುಭವಾಗಲಿ',
    'cheers': 'ಚಿಯರ್ಸ್',
    # ----------------- Common Words -----------------
    'i': 'ನಾನು',
    'you': 'ನೀವು',
    'we': 'ನಾವು',
    'my': 'ನನ್ನ',
    'your': 'ನಿಮ್ಮ',
    'name': 'ಹೆಸರು',
    'is': '',
    'am': '',
    'love': 'ಪ್ರೇಮ',
    'friend': 'ಮಿತ್ರ',
    'family': 'ಕುಟುಂಬ',
    'home': 'ಮನೆ',
    'school': 'ಶಾಲೆ',
    'work': 'ಕೆಲಸ',
    'food': 'ಆಹಾರ',
    'water': 'ನೀರು',
    'book': 'ಪುಸ್ತಕ',
    'pen': 'ಪೆನ್',
    'swathi': "ಸ್ವಾತಿ",
    'computer': 'ಕಂಪ್ಯೂಟರ್',
    'phone': 'ಫೋನ್',
    'car': 'ಕಾರು',
    'bus': 'ಬಸ್',
    'train': 'ಟ್ರೆయిన్',
    'city': 'ನಗರ',
    'village': 'ಗ್ರಾಮ',
}


def translate_to_kannada(sentence):
    words = sentence.lower().split()
    translated_words = [KAN_DICT.get(word, word) for word in words]
    return ' '.join(translated_words)

# ==================== PREPROCESSING ====================
def preprocess_for_model(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (64,64))
    normalized = resized / 255.0
    final = np.expand_dims(normalized, axis=-1)
    final = np.expand_dims(final, axis=0)
    return final

# ==================== LOAD MODEL ====================
if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(str(MODEL_PATH))

with open(LABELS_PATH,'r') as f:
    labels = json.load(f)

# ==================== MEDIA PIPE ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ==================== CAMERA ====================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ==================== VARIABLES ====================
pred_buffer = deque(maxlen=BUFFER_SIZE)
confidence_buffer = deque(maxlen=BUFFER_SIZE)
last_stable_sign = None
stable_count = 0
current_word = ''
sentence_text = ''
pending_letter = None

if not SENTENCE_FILE.exists():
    SENTENCE_FILE.touch()

fps_times = deque(maxlen=30)
last_time = time.time()
frame_count = 0
detection_count = 0
translator = Translator()

# ==================== UTILITY FUNCTIONS ====================
def calculate_fps(last_time, fps_times):
    current_time = time.time()
    fps_times.append(1.0 / (current_time - last_time) if current_time > last_time else 0)
    last_time = current_time
    fps = sum(fps_times)/len(fps_times) if fps_times else 0
    return fps, last_time

def append_word_to_file(word):
    global sentence_text
    sentence_text += (word + ' ')
    with open(SENTENCE_FILE, 'a') as f:
        f.write(word + '\n')

# ==================== MAIN LOOP ====================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    h,w = frame.shape[:2]
    frame_count += 1
    fps, last_time = calculate_fps(last_time, fps_times)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_sign = None
    current_confidence = 0
    hand_bbox = None

    if result.multi_hand_landmarks:
        detection_count += 1
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Centered square ROI
            cx = int(np.mean(x_coords)*w)
            cy = int(np.mean(y_coords)*h)
            size = int(max(max(x_coords)-min(x_coords), max(y_coords)-min(y_coords))*w*1.5)
            x_min = max(cx - size//2, 0)
            y_min = max(cy - size//2, 0)
            x_max = min(cx + size//2, w)
            y_max = min(cy + size//2, h)
            hand_bbox = (x_min, y_min, x_max, y_max)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                preprocessed = preprocess_for_model(roi)
                prediction = model.predict(preprocessed, verbose=0)
                current_confidence = float(np.max(prediction))
                predicted_idx = np.argmax(prediction)
                current_sign = labels[predicted_idx]
                pred_buffer.append(current_sign)
                confidence_buffer.append(current_confidence)

    # Smooth predictions
    final_sign = None
    if pred_buffer and confidence_buffer:
        counter = Counter(pred_buffer)
        most_common, count = counter.most_common(1)[0]
        agreement = count/len(pred_buffer)
        avg_conf = sum([conf for s, conf in zip(pred_buffer, confidence_buffer) if s==most_common])/count
        if agreement >= MIN_AGREEMENT and avg_conf >= CONFIDENCE_THRESHOLD:
            final_sign = most_common
            if final_sign == last_stable_sign:
                stable_count += 1
            else:
                stable_count = 1
                last_stable_sign = final_sign
        else:
            stable_count = 0

    # Pending letter for manual approval
    if stable_count >= 5 and final_sign:
        pending_letter = final_sign
    else:
        pending_letter = None

    # Draw hand bbox
    if hand_bbox:
        cv2.rectangle(frame, (hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3]), COLOR_PRIMARY, 3)

    # Draw info panel
    cv2.rectangle(frame,(0,0),(w,140),COLOR_PANEL,-1)
    status_text = f"Current Word: {current_word} | Sentence: {sentence_text}"
    cv2.putText(frame, status_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

    if pending_letter:
        cv2.putText(frame,f"Detected Letter: {pending_letter} (A=Accept, X=Skip)",(10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,COLOR_PRIMARY,2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,COLOR_TEXT,2)
    cv2.imshow("Hand Sign Recognition - Manual Letter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a') and pending_letter:
        current_word += pending_letter
        print(f"[LETTER ACCEPTED] {pending_letter}")
        pending_letter = None
    elif key == ord('x') and pending_letter:
        print(f"[LETTER SKIPPED] {pending_letter}")
        pending_letter = None
    elif key == ord(' '):
        if current_word:
            append_word_to_file(current_word)
            print(f"[WORD SAVED] -> {current_word}")
            current_word = ''

cap.release()
cv2.destroyAllWindows()

# ==================== TRANSLATION ====================
print("\n✅ Final Sentence in English:")
print(sentence_text.strip())

# Hindi translation
translated_hi = translator.translate(sentence_text.strip(), src='en', dest='hi').text
print("\n✅ Final Sentence in Hindi:")
print(translated_hi)

# Kannada translation
translated_ka = translator.translate(sentence_text.strip(), src='en', dest='kn').text
print("\n✅ Final Translator Sentence in kannada:")
print(translated_ka)

# Kannada translation
translated_te = translator.translate(sentence_text.strip(), src='en', dest='te').text
print("\n✅ Final Translator Sentence in telugu :")
print(translated_te)

