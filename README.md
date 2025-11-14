# 🤚 Hand Sign Recognition System

A real-time hand sign recognition system using deep learning and computer vision. This project uses MediaPipe for hand detection and a custom CNN model for sign classification.

---

## ✨ Features

- 🎯 **High Accuracy**: ~97% accuracy on test set
- 🚀 **Real-time Detection**: Smooth 30 FPS performance
- 🎨 **Modern UI**: Clean, informative interface with live feedback
- 🔧 **Robust Preprocessing**: CLAHE enhancement for varying lighting conditions
- 📊 **Smart Smoothing**: Prediction buffering for stable results
- 🎮 **Interactive Controls**: Clear buffer, reset stats, save screenshots

---

## 📁 Project Structure

```
HAND-GESTURE-RECOGNITION/
│
├── data/                          # Original dataset
│   ├── train/                     # Training images
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── test/                      # Test images
│
├── data_grayscale/               # Converted grayscale dataset (auto-generated)
│   ├── train/
│   └── test/
│
├── model/                        # Trained models and metadata
│   ├── hand_sign_model.h5       # Final trained model
│   ├── best_hand_sign_model.h5  # Best checkpoint
│   ├── class_labels.json        # Class label mapping
│   └── training_history.json    # Training metrics
│
├── src/                          # Source code
│   ├── train_model.py           # Model training script
│   ├── gray_scale_model.py      # Dataset preprocessing
│   ├── recognize_real_time.py   # Real-time detection
│   └── utils.py                 # Utility functions
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd HAND-GESTURE-RECOGNITION

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your hand sign images in the following structure:

```
data/
├── train/
│   ├── A/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...
```

### 3. Convert to Grayscale (Optional but Recommended)

```bash
cd src
python gray_scale_model.py
```

This will create a `data_grayscale` folder with CLAHE-enhanced grayscale images for better performance.

### 4. Train Model

```bash
cd src
python train_model.py
```

**Training output:**
- `model/hand_sign_model.h5` - Final model
- `model/best_hand_sign_model.h5` - Best checkpoint
- `model/class_labels.json` - Label mapping
- `model/training_history.json` - Training metrics

### 5. Run Real-Time Detection

```bash
cd src
python recognize_real_time.py
```

---

## 🎮 Controls (Real-Time Detection)

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `C` | Clear prediction buffer |
| `R` | Reset all statistics |
| `S` | Save screenshot |

---

## 🎨 UI Features

The real-time detection window includes:

- **Live Prediction Display**: Large, clear sign prediction
- **Confidence Bar**: Visual confidence meter with color coding
  - 🟢 Green: High confidence (>80%)
  - 🟡 Yellow: Medium confidence (60-80%)
  - 🟠 Orange: Low confidence (<60%)
- **Stability Indicator**: Shows if prediction is stable
- **Hand Landmarks**: MediaPipe skeleton overlay
- **Smart Bounding Box**: Adaptive hand tracking
- **Performance Metrics**: FPS, frame count, detection rate
- **Status Messages**: Clear feedback on system state

---

## 🔧 Configuration

### Model Training (`train_model.py`)

```python
IMG_SIZE = 64                    # Image dimensions
BATCH_SIZE = 32                  # Training batch size
EPOCHS = 50                      # Maximum epochs
LEARNING_RATE = 0.001           # Initial learning rate
VALIDATION_SPLIT = 0.2          # Train/val split ratio
```

### Real-Time Detection (`recognize_real_time.py`)

```python
CONFIDENCE_THRESHOLD = 0.60     # Min confidence to show prediction
BUFFER_SIZE = 10                # Smoothing buffer size
MIN_AGREEMENT = 0.40            # Min agreement in buffer (40%)
FRAME_WIDTH = 1280              # Camera resolution width
FRAME_HEIGHT = 720              # Camera resolution height
```

---

## 📊 Model Architecture

```
Input (64x64x1 grayscale)
    ↓
[Conv Block 1] → 32 filters → BatchNorm → Dropout(0.25)
    ↓
[Conv Block 2] → 64 filters → BatchNorm → Dropout(0.25)
    ↓
[Conv Block 3] → 128 filters → BatchNorm → Dropout(0.25)
    ↓
[Dense Layers] → 256 → 128 → Softmax
    ↓
Output (N classes)
```

**Key Features:**
- Batch normalization for stable training
- Dropout for regularization
- Data augmentation for robustness
- CLAHE preprocessing for lighting invariance

---

## 🛠️ Troubleshooting

### Model not recognizing signs correctly

1. **Check preprocessing consistency**: Ensure training and inference use same preprocessing
2. **Verify model path**: Confirm `best_hand_sign_model.h5` exists
3. **Check lighting**: System works best with good, even lighting
4. **Hold sign steady**: Keep hand still for 1-2 seconds
5. **Adjust thresholds**: Lower `CONFIDENCE_THRESHOLD` if needed

### Low FPS

1. **Reduce resolution**: Lower `FRAME_WIDTH` and `FRAME_HEIGHT`
2. **Use GPU**: Install `tensorflow-gpu` if you have CUDA
3. **Optimize buffer**: Reduce `BUFFER_SIZE`

### Camera not opening

1. **Check camera index**: Try different values for `CAMERA_INDEX` (0, 1, 2...)
2. **Permission issues**: Ensure camera permissions are granted
3. **Test camera**: Use `cv2.VideoCapture(0).isOpened()` to verify

---

## 📈 Performance Tips

1. **Dataset Quality**: More diverse training data = better accuracy
2. **Lighting**: Train with varied lighting conditions
3. **Hand Position**: Keep hand centered in frame
4. **Background**: Simple, contrasting background works best
5. **Distance**: Maintain consistent distance from camera

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **TensorFlow/Keras**: Deep learning framework
- **MediaPipe**: Hand detection and tracking
- **OpenCV**: Computer vision operations

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for accessible hand sign recognition**