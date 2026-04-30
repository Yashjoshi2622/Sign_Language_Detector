# Advanced Sign Language Studio

![GitHub Workflow Status](https://github.com/Yashjoshi2622/Sign_Language_Detector/actions/workflows/python-app.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

A polished Streamlit app for sign language gesture recognition with webcam capture, live prediction, dataset creation, model training, and sentence builder support.

## 🚀 Overview

This repository demonstrates a complete local sign language solution:

- Webcam capture and image upload for gesture input
- Hand landmark extraction using MediaPipe
- Live prediction using heuristic and trained models
- Few-shot learning with sample collection
- Synthetic bootstrap dataset generation for quick startup
- Model training and evaluation with ensemble classification
- Confidence trend and sentence building features

## ✨ Features

- Real-time Streamlit interface with camera and upload workflows
- Number sign mode for digits `0-9`
- Custom label collection and adaptive training
- Confidence smoothing, cooldown, and smart lock logic
- Training dashboard with holdout accuracy and confusion matrix
- Model persistence in `model/`
- Optional text-to-speech support

## 🧪 Prerequisites

- Python 3.10+
- Windows is recommended for webcam support
- Virtual environment (recommended)

## 📥 Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to launch the UI.

## 🧠 Quick Start Without Data

If you do not yet have gesture samples, generate a synthetic starter dataset and train the model:

```bash
python bootstrap_dataset.py
python train_lstm.py
```

Then run the app again and it will use the trained model automatically.

## 🔧 Train Your Custom Model

Collect at least 10 samples across at least two labels, then run:

```bash
python train_lstm.py
```

This writes:

- `model/gesture_rf.joblib`
- `model/labels.json`

## 📚 Project Structure

- `app.py` — main Streamlit app and interface logic
- `train_lstm.py` — classifier training and model serialization
- `bootstrap_dataset.py` — synthetic dataset generator for bootstrapping
- `data/embeddings.json` — saved embeddings and sample labels
- `model/` — saved trained model and label mapping
- `requirements.txt` — Python dependency list
- `.gitignore` — ignored files and directories

## 🧠 How to Use

### Start predicting

1. Capture or upload a hand sign image
2. Choose `Number signs (0-9)` or `Custom labels`
3. Save samples for each label
4. Train the model after collecting enough data
5. Use the live webcam mode for continuous prediction

### Number sign mode

- Select digits `0-9`
- Capture or upload a gesture
- Save examples per digit
- Train the model for stable digit recognition

### Real-time webcam mode

- Enable `Run continuous detection`
- Set the duration and FPS
- Choose prediction mode
- Enable smoothing and cooldown
- Use the smart lock for stable sentence output

## ⚠️ Troubleshooting

### Protobuf compatibility

If you encounter an error like:

`cannot import name 'runtime_version' from 'google.protobuf'`

Install a compatible protobuf version:

```bash
pip install protobuf==4.23.4
```

### scikit-learn unpickle warning

If a warning appears during model load, it means the saved model was created with a newer scikit-learn version than the runtime. Retraining the model in the current environment will resolve the warning.

## 📌 Notes

- Synthetic data is for demo use only.
- For best accuracy, collect real samples under varied lighting and angles.
- Use the training dashboard to inspect model performance and confusion.

## 🤝 Contribution

Contributions, bug reports, and improvements are welcome. Feel free to open issues or submit pull requests.

## 📜 License

This project is released under the MIT License.
