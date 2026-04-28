# Advanced Sign Language Studio

This project is a Streamlit-based sign language detection app designed to work even when you do not have an initial dataset.

## What You Get

- Capture hand sign image directly from webcam using Streamlit camera input
- Upload hand sign images from your system
- Real-time webcam stream mode for continuous prediction
- Live confidence trend mini-graph (raw vs smoothed confidence + lock state)
- Auto hand landmark extraction with MediaPipe
- No-dataset startup mode using heuristic gesture prediction
- Few-shot learning by saving your own labeled samples
- Dedicated number-sign mode for digits 0-9
- Optional model training with augmentation + ensemble classifier
- Training dashboard with per-label counts, holdout accuracy, confusion matrix, and misclassification table
- Sentence builder with confidence gating and text-to-speech

## Project Files

- `app.py`: Streamlit UI for capture/upload, sample collection, prediction, sentence builder
- `train_lstm.py`: Trains a RandomForest classifier using collected samples (`data/embeddings.json`)
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## How to Use Without Dataset

1. Open app and use camera capture or upload image.
2. Choose `Number signs (0-9)` if you want digit recognition, or `Custom labels` for your own classes.
3. Save samples for each digit or custom label.
4. Start predicting immediately with built-in few-shot kNN.
5. For better accuracy, train a model after collecting enough samples.

## One-Click Real Sample Collection

1. Capture or upload a clear hand sign image.
2. Set `Samples to collect at once` in the sidebar.
3. Click `Collect real samples now`.
4. The app saves the original embedding plus augmented variants into the selected label.
5. Retrain the model after collecting a few batches.

## Number Sign Mode

1. Select `Number signs (0-9)` in the sidebar.
2. Capture or upload a hand gesture for a digit.
3. Save the sample under the chosen digit label.
4. Train with `python train_lstm.py` after collecting enough samples.
5. The app will then use the trained model for digit prediction.

## Real-time Webcam Mode

1. Scroll to `Real-time Webcam Stream`.
2. Enable `Run continuous detection`.
3. Set duration and FPS.
4. Choose `Only predict` or `Predict + auto add to sentence`.
5. Enable smoothing for more stable labels.
6. Enable temporal cooldown to prevent repeated same-label spam.
7. Enable adaptive cooldown so low-confidence predictions automatically get longer cooldown.
8. Enable confidence-trend smart lock to temporarily hold sentence updates during unstable frame bursts.
9. Tune smart lock window/std/switch threshold and hold seconds.
10. Enable live confidence trend graph for easier tuning.
11. The app processes live frames continuously for the selected duration.

## Train Your Custom Model

Collect at least 10 samples across at least 2 labels, then run:

```bash
python train_lstm.py
```

If you do not have any data yet, you can generate a synthetic starter number dataset (0-9):

```bash
python bootstrap_dataset.py
python train_lstm.py
```

Note: Synthetic data is only for bootstrapping/demo. For real accuracy, replace it with your own camera/upload samples.

The app shows a startup banner that tells you whether it is using bootstrap synthetic data or your custom samples.

This saves:

- `model/gesture_rf.joblib`
- `model/labels.json`

After that, the app automatically uses the trained model for prediction.

Training now includes:

- Embedding augmentation (noise + scale jitter)
- Soft-voting tree ensemble (RandomForest + ExtraTrees)
- Better class balancing for imbalanced labels

These typically improve practical accuracy and reduce jitter over a single baseline model.

## Accuracy Improvement Tips

1. Use balanced samples per label.
2. Capture each sign under slightly different angles and distances.
3. Keep lighting stable and hand clearly visible.
4. Retrain after adding new samples.
5. Keep smoothing, adaptive cooldown, and smart lock enabled during realtime inference.
6. Use Training Dashboard after retraining to inspect confusion matrix and fix weak labels.

## Advanced Features You Can Add Next

- Multi-hand support for two-handed signs
- Real-time video stream inference with frame smoothing
- Sequence model (LSTM/Transformer) for dynamic signs and words
- Language translation layer (sign-to-text in local language)
- Cloud sync of user-specific gesture profiles
- Accuracy dashboard and confusion matrix in Streamlit
