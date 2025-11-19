"""
Hand Sign Recognition - Simple Training Script
Trains once and saves model - won't retrain if model exists
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
from pathlib import Path

# ==================== CONFIGURATION ====================
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ==================== PATHS ====================
BASE_DIR = Path(r"C:\SignLang\Gesture")
TRAIN_DIR = BASE_DIR / 'data_grayscale' / 'train'
TEST_DIR = BASE_DIR / 'data_grayscale' / 'test'
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'hand_sign_model.h5'

MODEL_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🚀 HAND SIGN RECOGNITION - TRAINING")
print("=" * 80)

# Check if model already exists
continue_training = False

if MODEL_PATH.exists():
    print(f"\n🔁 Existing model found at: {MODEL_PATH}")
    response = input("Do you want to CONTINUE training the existing model? (y/n): ")

if MODEL_PATH.exists():
    print(f"\n🔁 Existing model found at: {MODEL_PATH}")
    response = input("Do you want to CONTINUE training the existing model? (y/n): ")

    if response.lower() == 'y':
        print("📥 Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # 🔥 IMPORTANT: Re-compile the loaded model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✅ Model loaded & compiled. Training will continue...\n")
        continue_training = True

    else:
        print("🆕 Starting fresh training...")
        continue_training = False

# Verify directories
if not TRAIN_DIR.exists():
    print(f"\n❌ Training directory not found: {TRAIN_DIR}")
    exit(1)

print(f"\n📁 Training data: {TRAIN_DIR}")
print(f"📁 Test data: {TEST_DIR}")
print(f"💾 Model output: {MODEL_PATH}")

# ==================== DATA LOADING ====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Classes: {NUM_CLASSES}")

# Save class labels
labels = sorted(train_generator.class_indices.keys(), 
                key=lambda x: train_generator.class_indices[x])

with open(MODEL_DIR / 'class_labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

print(f"✅ Classes: {', '.join(labels)}")

# ==================== BUILD MODEL ====================
if not continue_training:
    print("\n🏗️ Building improved model...")

    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# ==================== TRAINING ====================
print("\n🎯 Starting training...")

# 1️⃣ Define lr_scheduler FIRST
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,         # reduce LR by 70%
    patience=4,         # wait 4 epochs without improvement
    min_lr=1e-6,        # never go below this
    verbose=1
)

# 2️⃣ Add callbacks
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    lr_scheduler
]

# 3️⃣ Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n" + "=" * 80)
print("✅ Training Complete!")
print(f"💾 Model saved: {MODEL_PATH}")
print("\n📝 Next step: Run 'python recognize.py' to test real-time recognition")
print("=" * 80)