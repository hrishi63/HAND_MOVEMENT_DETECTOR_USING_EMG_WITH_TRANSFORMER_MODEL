# 🧠 EMG Hand Movement Detector

Real-time hand gesture detection system using electromyography (EMG) signals, powered by a hybrid CNN-Transformer deep learning model.

![System Demo](demo.gif) <!-- Add a demo GIF/screenshot if you have one -->

## 🌟 Features

- **Real-time EMG Signal Processing**: 500Hz sampling with adaptive baseline RMS detection
- **Dual Detection System**:
  - 🔴 **RMS-based Detection**: Fast, threshold-based clench detection with hysteresis
  - 🧠 **AI-Powered Analysis**: Deep learning model (CNN + Transformer) for gesture classification
- **Live Visualization**: Real-time waveform display with color-coded hand states
- **6 Real-time Metrics**:
  - AI Gesture Confidence
  - Neural Activation
  - Signal Stability
  - Response Sharpness
  - Muscle Consistency
  - Responsiveness
- **Web-Based Interface**: Built with React for smooth, interactive UI

## 🏗️ System Architecture
```
Arduino (EMG Sensor) → Serial (500Hz) → Python Backend → WebSocket → React Frontend
                                              ↓
                                        AI Model (ONNX)
```

### Components:
1. **Hardware**: MyoWare EMG sensor + Arduino Uno
2. **Backend**: Python Flask server with real-time signal processing
3. **AI Model**: Hybrid CNN-Transformer (PyTorch → ONNX)
4. **Frontend**: React app with ONNX Runtime Web for in-browser inference

## 📊 Model Architecture

**Hybrid CNN + Transformer** (400K parameters)
```
Input (256 samples, 0.5s window)
    ↓
CNN Feature Extractor (2 Conv blocks)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2 layers, 8 heads)
    ↓
Classification Head
    ↓
Output (Gesture probability 0-1)
```

**Performance**: 75.8% validation accuracy on 13 subjects, 13,000 samples

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Arduino IDE
- MyoWare EMG Sensor

### 1. Hardware Setup

1. Connect MyoWare sensor to Arduino:
   - Signal → A0
   - VCC → 5V
   - GND → GND

2. Upload Arduino code:
```bash
cd arduino
# Open emg_reader.ino in Arduino IDE and upload
```

3. Note your Arduino's serial port (e.g., `/dev/ttyUSB0` or `COM3`)

### 2. Backend Setup
```bash
# Clone repository
git clone https://github.com/yourusername/emg-hand-detector.git
cd emg-hand-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Update serial port in server.py (line 12)
# ser = serial.Serial('/dev/ttyUSB0', 115200)

# Start backend server
python server.py
```

Server will run on `http://localhost:5000`

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Copy model files to public folder
cp ../models/emg_transformer_web.onnx public/model/
cp ../models/emg_transformer_web.onnx.data public/model/

# Start development server
npm start
```

App will open at `http://localhost:3000`

## 🎮 Usage

1. **Attach EMG sensor** to your forearm (muscle belly)
2. **Calibrate** by relaxing for 3 seconds
3. **Click Play** ▶️ to start monitoring
4. **Clench your fist** and watch real-time metrics update
5. **View AI analysis** in the bottom panel

### Controls:
- **▶️ Play/⏸️ Pause**: Start/stop real-time monitoring
- **Gesture Cycles**: Counts open → clenched transitions

## 📁 Project Structure
```
emg-hand-detector/
├── arduino/
│   └── emg_reader.ino          # Arduino firmware
├── backend/
│   ├── server.py               # Flask server with signal processing
│   ├── train_model.py          # Model training script
│   ├── model.py                # Model architecture definition
│   └── export_working.py       # ONNX export script
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── EMGHandVisualizer.jsx  # Main visualizer
│   │   │   └── AIAnalyzer.jsx         # AI metrics panel
│   │   └── App.js
│   └── public/
│       └── model/              # ONNX model files
├── models/
│   ├── emg_transformer_web.onnx
│   └── emg_transformer_web.onnx.data
├── dataset/                     # Training data (not included)
├── requirements.txt
└── README.md
```

## 🧪 Training Your Own Model

### 1. Collect Data
```bash
# Run data collection script (create your own based on needs)
python collect_data.py
```

Dataset format:
- `X0.npy` to `X12.npy`: EMG windows (shape: [N, 256])
- `Y0.npy` to `Y12.npy`: Labels (0=open, 1=clenched)

### 2. Train Model
```bash
python train_model.py
```

Training config:
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Augmentation: Random scaling, noise, time shifts
- Loss: Binary Cross-Entropy

Output: `emg_transformer_real.pth`

### 3. Export to ONNX
```bash
python export_working.py
```

Creates: `emg_transformer_web.onnx` + `emg_transformer_web.onnx.data`

## 📈 Metrics Explained

| Metric | Calculation | Meaning |
|--------|-------------|---------|
| **AI Confidence** | Model output aligned with RMS state | AI's certainty in gesture (70-100% is good) |
| **Neural Activation** | Mean absolute amplitude / 30 × 100 | Overall muscle activity level |
| **Signal Stability** | 100 - (CV × 50) | Signal consistency (low tremor) |
| **Response Sharpness** | Peak-to-peak amplitude / 100 × 100 | Strength of muscle activation |
| **Muscle Consistency** | 100 / (1 + variance/10) | Pattern repeatability |
| **Responsiveness** | Mean derivative × 20 | Speed of muscle response |

**Color coding:**
- 🟢 Green (70-100%): Excellent
- 🟠 Orange (40-70%): Moderate
- 🔴 Red (0-40%): Poor

## 🔧 Troubleshooting

### Model won't load in browser
- Ensure both `.onnx` and `.onnx.data` files are in `public/model/`
- Check browser console for specific errors
- Verify files are accessible at `http://localhost:3000/model/emg_transformer_web.onnx`

### Serial connection fails
- Check Arduino is connected and port is correct
- Try different baud rates (115200 recommended)
- On Linux, add user to `dialout` group: `sudo usermod -a -G dialout $USER`

### Poor detection accuracy
- Ensure sensor is on muscle belly (not tendon)
- Clean skin with alcohol before application
- Adjust RMS threshold in `server.py` (line 28)
- Recalibrate by relaxing hand for 3 seconds

### High latency
- Reduce Arduino serial buffer if needed
- Check CPU usage (transformer inference can be heavy)
- Lower frontend polling rate in AIAnalyzer.jsx (line 171)

## 🛠️ Technologies Used

- **Hardware**: MyoWare EMG Sensor, Arduino Uno
- **Backend**: Python, Flask, PySerial, PyTorch, ONNX
- **Frontend**: React, ONNX Runtime Web
- **ML**: PyTorch (CNN + Transformer architecture)

## 📝 Technical Details

### Signal Processing Pipeline:
1. **Analog filtering** (Arduino): Bandpass filter + envelope detection
2. **Digital sampling**: 500Hz via serial
3. **RMS calculation**: Rolling window with adaptive baseline
4. **Hysteresis thresholding**: Prevents flickering
5. **AI inference**: 256-sample windows every 400ms

### Model Training:
- **Dataset**: 13 subjects, ~13,000 samples
- **Architecture**: Hybrid CNN-Transformer (400K parameters)
- **Training time**: ~30 min on GPU
- **Validation accuracy**: 75.8%

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset collected from 13 volunteer subjects
- Inspired by research in EMG-based HCI
- MyoWare sensor by Advancer Technologies

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle) - email@example.com

Project Link: [https://github.com/yourusername/emg-hand-detector](https://github.com/yourusername/emg-hand-detector)

---

⭐ **Star this repo if you found it useful!**
```

---

## **Optional: LICENSE (MIT)**

Create a `LICENSE` file:
```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
