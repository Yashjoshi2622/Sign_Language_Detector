import json
import os
import time
from collections import Counter, deque
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    # Preferred import path for newer MediaPipe package layouts.
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import hands as mp_hands
except Exception:
    mp_hands = None
    mp_drawing = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import joblib
except Exception:
    joblib = None


st.set_page_config(page_title="Sign Language Studio", layout="wide")

DATA_DIR = "data"
MODEL_DIR = "model"
DB_PATH = os.path.join(DATA_DIR, "embeddings.json")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_rf.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if mp_hands is None or mp_drawing is None:
    if hasattr(mp, "solutions"):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
    else:
        st.error(
            "MediaPipe Hands APIs are unavailable in this environment. "
            "Reinstall mediapipe in this virtual environment and restart Streamlit."
        )
        st.stop()


def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def bytes_to_bgr(uploaded):
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def normalize_landmarks(hand_landmarks):
    points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    points = points - points[0]
    scale = np.max(np.linalg.norm(points, axis=1)) + 1e-6
    points = points / scale
    return points.flatten()


def augment_embedding(embedding, rng, noise_std=0.015, scale_jitter=0.03):
    augmented = np.array(embedding, dtype=np.float32).copy()
    augmented = augmented.reshape(-1, 2)
    scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
    noise = rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    augmented = (augmented * scale) + noise
    augmented = augmented.reshape(-1)
    return np.clip(augmented, -2.0, 2.0).astype(np.float32)


def collect_samples_for_label(db, label, embedding, sample_count=20, include_original=True):
    label = label.strip().upper()
    if not label:
        return 0

    rng = np.random.default_rng()
    db.setdefault(label, [])

    added = 0
    if include_original:
        db[label].append(np.array(embedding, dtype=np.float32).tolist())
        added += 1

    while added < sample_count:
        db[label].append(augment_embedding(embedding, rng).tolist())
        added += 1

    save_db(db)
    return added


def detect_hand_and_embedding(image_bgr, min_conf=0.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    ) as hands:
        result = hands.process(image_rgb)

    annotated = image_bgr.copy()
    if not result.multi_hand_landmarks:
        return None, annotated

    hand_landmarks = result.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return normalize_landmarks(hand_landmarks), annotated


def heuristic_predict(embedding):
    points = embedding.reshape(21, 2)
    wrist = points[0]

    def extended(tip, pip):
        return points[tip][1] < points[pip][1]

    index_on = extended(8, 6)
    middle_on = extended(12, 10)
    ring_on = extended(16, 14)
    pinky_on = extended(20, 18)

    thumb_dist = np.linalg.norm(points[4] - wrist)
    thumb_on = thumb_dist > 0.32

    fingers = [thumb_on, index_on, middle_on, ring_on, pinky_on]
    count = int(sum(fingers))

    if count >= 4:
        return "OPEN_PALM", 0.62
    if count == 0:
        return "FIST", 0.64
    if index_on and middle_on and not ring_on and not pinky_on:
        return "PEACE", 0.66
    if index_on and not middle_on and not ring_on and not pinky_on:
        return "POINT", 0.60
    return "UNKNOWN_HAND_SIGN", 0.50


def flatten_db(db):
    vectors, labels = [], []
    for label, samples in db.items():
        for sample in samples:
            vectors.append(np.array(sample, dtype=np.float32))
            labels.append(label)
    if not vectors:
        return np.array([]), []
    return np.vstack(vectors), labels


def build_supervised_arrays(db, target_labels=None):
    X, y = [], []
    if target_labels is None:
        target_labels = sorted(db.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(target_labels)}

    for lbl, samples in db.items():
        if lbl not in label_to_idx:
            continue
        for sample in samples:
            X.append(np.array(sample, dtype=np.float32))
            y.append(label_to_idx[lbl])

    if not X:
        return None, None, target_labels
    return np.vstack(X), np.array(y), target_labels


def evaluate_model_holdout(clf, db, model_labels):
    X, y, labels = build_supervised_arrays(db, target_labels=model_labels)
    if X is None or y is None:
        return None, "No compatible labeled samples found for current trained model labels."

    class_counts = np.bincount(y, minlength=len(labels))
    if np.min(class_counts) < 2 or len(labels) < 2:
        return None, "Need at least 2 samples per label and at least 2 labels for stratified holdout evaluation."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    y_conf = np.max(y_proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)

    mis_rows = []
    for idx, (yt, yp) in enumerate(zip(y_test, y_pred)):
        if yt != yp:
            mis_rows.append(
                {
                    "sample_index": int(idx),
                    "true_label": labels[int(yt)],
                    "pred_label": labels[int(yp)],
                    "pred_confidence": float(y_conf[idx]),
                }
            )
    mis_df = pd.DataFrame(mis_rows)

    return {
        "accuracy": acc,
        "cm_df": cm_df,
        "report_df": report_df,
        "mis_df": mis_df,
        "test_samples": int(len(y_test)),
    }, None


def is_bootstrap_number_dataset(db):
    expected_labels = [str(i) for i in range(10)]
    if sorted(db.keys()) != expected_labels:
        return False
    counts = [len(db[label]) for label in expected_labels]
    return len(set(counts)) == 1 and counts[0] >= 50


def knn_predict(embedding, db, k=3):
    X, y = flatten_db(db)
    if len(y) == 0:
        return None, 0.0

    dists = np.linalg.norm(X - embedding, axis=1)
    idx = np.argsort(dists)[: min(k, len(dists))]
    nearest_labels = [y[i] for i in idx]
    nearest_dists = dists[idx]

    scores = {}
    for lbl, d in zip(nearest_labels, nearest_dists):
        scores[lbl] = scores.get(lbl, 0.0) + 1.0 / (d + 1e-6)

    best_label = max(scores, key=scores.get)
    confidence = scores[best_label] / (sum(scores.values()) + 1e-9)
    return best_label, float(confidence)


def load_trained_model():
    if not (joblib and os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, None
    clf = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return clf, labels


def model_predict(embedding, clf, labels):
    prob = clf.predict_proba([embedding])[0]
    idx = int(np.argmax(prob))
    return labels[idx], float(prob[idx])


def predict_label(embedding, clf, model_labels, db):
    pred_label, pred_conf = None, 0.0
    if clf is not None and model_labels is not None:
        pred_label, pred_conf = model_predict(embedding, clf, model_labels)
        mode_used = "Trained RandomForest"
    else:
        pred_label, pred_conf = knn_predict(embedding, db, k=5)
        mode_used = "Few-shot kNN"

    if pred_label is None:
        pred_label, pred_conf = heuristic_predict(embedding)
        mode_used = "No-dataset heuristic"

    return pred_label, pred_conf, mode_used


def smooth_prediction(history):
    if not history:
        return None, 0.0

    labels = [item[0] for item in history]
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    candidates = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        # Tie-break by average confidence in the history window.
        avg_conf = {}
        for lbl in candidates:
            confs = [conf for item_lbl, conf in history if item_lbl == lbl]
            avg_conf[lbl] = sum(confs) / max(1, len(confs))
        chosen = max(avg_conf, key=avg_conf.get)

    chosen_confs = [conf for item_lbl, conf in history if item_lbl == chosen]
    return chosen, float(sum(chosen_confs) / max(1, len(chosen_confs)))


def add_to_sentence(text, confidence, threshold):
    if confidence < threshold:
        return
    if not st.session_state.sentence or st.session_state.sentence[-1] != text:
        st.session_state.sentence.append(text)


def add_to_sentence_with_cooldown(text, confidence, threshold, cooldown_sec=0.0):
    if confidence < threshold:
        return

    now_ts = time.time()
    last_added_by_label = st.session_state.get("last_added_by_label", {})
    last_added_ts = float(last_added_by_label.get(text, 0.0))

    if cooldown_sec > 0 and (now_ts - last_added_ts) < cooldown_sec:
        return

    if not st.session_state.sentence or st.session_state.sentence[-1] != text:
        st.session_state.sentence.append(text)
        last_added_by_label[text] = now_ts
        st.session_state.last_added_by_label = last_added_by_label


def compute_adaptive_cooldown(base_cooldown, confidence, low_conf_cutoff, max_multiplier):
    if base_cooldown <= 0:
        return 0.0
    if confidence >= low_conf_cutoff:
        return float(base_cooldown)

    # Lower confidence increases cooldown linearly up to max_multiplier.
    normalized_gap = (low_conf_cutoff - confidence) / max(low_conf_cutoff, 1e-6)
    normalized_gap = min(max(normalized_gap, 0.0), 1.0)
    multiplier = 1.0 + normalized_gap * (max_multiplier - 1.0)
    return float(base_cooldown * multiplier)


def detect_unstable_trend(conf_history, label_history, conf_std_limit, switch_limit):
    if len(conf_history) < 4 or len(label_history) < 4:
        return False, "warming_up"

    conf_std = float(np.std(np.array(conf_history, dtype=np.float32)))
    switches = sum(1 for i in range(1, len(label_history)) if label_history[i] != label_history[i - 1])

    if conf_std >= conf_std_limit or switches >= switch_limit:
        reason = f"std={conf_std:.3f}, switches={switches}"
        return True, reason

    return False, f"std={conf_std:.3f}, switches={switches}"


def speak(text):
    if not pyttsx3:
        st.warning("Text-to-speech is unavailable. Install pyttsx3 to enable it.")
        return
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
NUMBER_LABELS = [str(i) for i in range(10)]
NUMBER_WORDS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}


if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_added_by_label" not in st.session_state:
    st.session_state.last_added_by_label = {}

if "smart_lock_until" not in st.session_state:
    st.session_state.smart_lock_until = 0.0

st.title("Sign Language Detection Studio")
st.caption("Capture from camera, upload images, build your own dataset, and predict gestures without a pre-existing dataset.")

db = load_db()
clf, model_labels = load_trained_model()

if db:
    if is_bootstrap_number_dataset(db):
        st.warning(
            "Bootstrap number dataset is active. Number prediction will work, but real camera samples will improve accuracy further."
        )
    else:
        st.success("Custom dataset loaded. Use the Training Dashboard to inspect accuracy and weak labels.")
else:
    st.info("No samples found yet. Use the built-in bootstrap or collect real samples from camera/upload.")

with st.sidebar:
    st.header("Dataset Builder")
    mode = st.radio("Gesture mode", ["Custom labels", "Number signs (0-9)"], index=1)
    if mode == "Number signs (0-9)":
        label_name = st.selectbox("Label name", NUMBER_LABELS, index=1)
        st.info("Number mode is active. Save and train using digits 0-9.")
    else:
        label_name = st.text_input("Label name", value="HELLO")
    real_sample_count = st.slider("Samples to collect at once", 5, 50, 20, 1)
    conf_threshold = st.slider("Sentence confidence threshold", 0.3, 0.99, 0.65, 0.01)
    st.write("Samples per label")
    sample_stats = {k: len(v) for k, v in db.items()}
    st.json(sample_stats if sample_stats else {"No labels yet": 0})

    if st.button("Clear all saved samples", type="secondary"):
        save_db({})
        st.success("All saved samples cleared.")

left, right = st.columns(2)

with left:
    st.subheader("Capture on Spot")
    cam_file = st.camera_input("Take a hand sign picture")

with right:
    st.subheader("Upload Image")
    upload_file = st.file_uploader("Upload a hand sign image", type=["png", "jpg", "jpeg"])

source_file = cam_file if cam_file is not None else upload_file

if source_file is not None:
    source_file = BytesIO(source_file.getvalue())
    image = bytes_to_bgr(source_file)
    if image is None:
        st.error("Unable to decode image. Try another image.")
    else:
        embedding, annotated = detect_hand_and_embedding(image)
        col_a, col_b = st.columns(2)
        col_a.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        col_b.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Hand", use_container_width=True)

        if embedding is None:
            st.warning("No hand detected. Ensure your full hand is visible and well-lit.")
        else:
            add_sample_col, infer_col, sentence_col = st.columns(3)
            if add_sample_col.button("Save sample to label"):
                db.setdefault(label_name.strip().upper(), []).append(embedding.tolist())
                save_db(db)
                st.success(f"Sample added to label: {label_name.strip().upper()}")

            if st.button("Collect real samples now"):
                collected = collect_samples_for_label(
                    db,
                    label_name,
                    embedding,
                    sample_count=real_sample_count,
                    include_original=True,
                )
                st.success(f"Collected {collected} samples for label: {label_name.strip().upper()}")

            pred_label, pred_conf, mode_used = predict_label(embedding, clf, model_labels, db)

            st.session_state.last_prediction = (pred_label, pred_conf, mode_used)

            if infer_col.button("Add prediction to sentence"):
                add_to_sentence(pred_label, pred_conf, conf_threshold)

            if sentence_col.button("Clear sentence"):
                st.session_state.sentence = []

st.subheader("Real-time Webcam Stream")
rt_col1, rt_col2 = st.columns(2)
with rt_col1:
    run_realtime = st.checkbox("Run continuous detection")
with rt_col2:
    realtime_mode = st.radio("Realtime action", ["Only predict", "Predict + auto add to sentence"], index=1)

duration_sec = st.slider("Realtime duration (seconds)", 5, 60, 15, 1)
fps = st.slider("Realtime FPS", 2, 15, 6, 1)
smoothing_on = st.checkbox("Enable smoothing (majority vote)", value=True)
smoothing_window = st.slider("Smoothing window size", 3, 9, 5, 1)
cooldown_on = st.checkbox("Enable temporal cooldown", value=True)
cooldown_sec = st.slider("Cooldown per label (seconds)", 0.0, 5.0, 1.2, 0.1)
adaptive_cooldown_on = st.checkbox("Enable adaptive cooldown", value=True)
adaptive_low_conf_cutoff = st.slider("Adaptive low-confidence cutoff", 0.50, 0.95, 0.75, 0.01)
adaptive_max_multiplier = st.slider("Adaptive max cooldown multiplier", 1.0, 5.0, 2.5, 0.1)
smart_lock_on = st.checkbox("Enable confidence-trend smart lock", value=True)
smart_lock_window = st.slider("Smart lock trend window", 4, 15, 7, 1)
smart_lock_conf_std_limit = st.slider("Smart lock confidence std limit", 0.02, 0.30, 0.10, 0.01)
smart_lock_switch_limit = st.slider("Smart lock label-switch limit", 1, 8, 3, 1)
smart_lock_hold_sec = st.slider("Smart lock hold seconds", 0.2, 3.0, 1.0, 0.1)
show_conf_graph = st.checkbox("Show live confidence trend graph", value=True)
graph_window = st.slider("Confidence graph window", 10, 120, 40, 1)

if run_realtime:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam for real-time mode.")
    else:
        frame_slot = st.empty()
        info_slot = st.empty()
        progress_slot = st.progress(0)
        graph_slot = st.empty()
        prediction_history = deque(maxlen=smoothing_window)
        trend_conf_history = deque(maxlen=smart_lock_window)
        trend_label_history = deque(maxlen=smart_lock_window)
        raw_conf_plot = deque(maxlen=graph_window)
        smooth_conf_plot = deque(maxlen=graph_window)
        lock_plot = deque(maxlen=graph_window)
        total_frames = max(1, duration_sec * fps)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue

            embedding, annotated = detect_hand_and_embedding(frame, min_conf=0.6)
            view_frame = frame
            if embedding is not None:
                pred_label, pred_conf, mode_used = predict_label(embedding, clf, model_labels, db)
                final_label, final_conf = pred_label, pred_conf
                active_cooldown = 0.0
                if smoothing_on:
                    prediction_history.append((pred_label, pred_conf))
                    smoothed_label, smoothed_conf = smooth_prediction(prediction_history)
                    if smoothed_label is not None:
                        final_label, final_conf = smoothed_label, smoothed_conf

                raw_conf_plot.append(float(pred_conf))
                smooth_conf_plot.append(float(final_conf))

                st.session_state.last_prediction = (final_label, final_conf, mode_used)

                trend_conf_history.append(final_conf)
                trend_label_history.append(final_label)
                now_ts = time.time()
                smart_lock_active = now_ts < float(st.session_state.smart_lock_until)
                unstable, trend_reason = detect_unstable_trend(
                    trend_conf_history,
                    trend_label_history,
                    smart_lock_conf_std_limit,
                    smart_lock_switch_limit,
                )
                if smart_lock_on and unstable:
                    st.session_state.smart_lock_until = max(
                        float(st.session_state.smart_lock_until),
                        now_ts + smart_lock_hold_sec,
                    )
                    smart_lock_active = True
                lock_remaining = max(0.0, float(st.session_state.smart_lock_until) - now_ts)
                lock_plot.append(1.0 if smart_lock_active else 0.0)

                if realtime_mode == "Predict + auto add to sentence":
                    active_cooldown = cooldown_sec if cooldown_on else 0.0
                    if cooldown_on and adaptive_cooldown_on:
                        active_cooldown = compute_adaptive_cooldown(
                            active_cooldown,
                            final_conf,
                            adaptive_low_conf_cutoff,
                            adaptive_max_multiplier,
                        )
                    if not smart_lock_active:
                        add_to_sentence_with_cooldown(
                            final_label,
                            final_conf,
                            conf_threshold,
                            cooldown_sec=active_cooldown,
                        )

                view_frame = annotated
                cv2.putText(
                    view_frame,
                    f"{final_label} ({final_conf:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                if smoothing_on:
                    if smart_lock_active:
                        info_slot.warning(
                            f"Raw: {pred_label} ({pred_conf:.2f}) | Smoothed: {final_label} ({final_conf:.2f}) | Cooldown: {active_cooldown:.2f}s | LOCKED {lock_remaining:.2f}s ({trend_reason})"
                        )
                    else:
                        info_slot.info(
                            f"Raw: {pred_label} ({pred_conf:.2f}) | Smoothed: {final_label} ({final_conf:.2f}) | Cooldown: {active_cooldown:.2f}s | Stable ({trend_reason})"
                        )
                else:
                    if smart_lock_active:
                        info_slot.warning(
                            f"Realtime prediction: {final_label} | confidence: {final_conf:.2f} | Cooldown: {active_cooldown:.2f}s | LOCKED {lock_remaining:.2f}s ({trend_reason})"
                        )
                    else:
                        info_slot.info(
                            f"Realtime prediction: {final_label} | confidence: {final_conf:.2f} | Cooldown: {active_cooldown:.2f}s | Stable ({trend_reason})"
                        )

                if show_conf_graph and len(raw_conf_plot) >= 2:
                    graph_slot.line_chart(
                        {
                            "raw_conf": list(raw_conf_plot),
                            "smoothed_conf": list(smooth_conf_plot),
                            "lock_state": list(lock_plot),
                        },
                        use_container_width=True,
                    )
            else:
                info_slot.warning("No hand detected in current frame.")

            frame_slot.image(cv2.cvtColor(view_frame, cv2.COLOR_BGR2RGB), caption="Realtime Webcam", use_container_width=True)
            progress_slot.progress(int(((i + 1) / total_frames) * 100))
            time.sleep(1.0 / max(1, fps))

        cap.release()
        st.success("Realtime detection completed.")

sentence = " ".join(st.session_state.sentence)

st.subheader("Prediction")
if st.session_state.last_prediction:
    label, conf, mode = st.session_state.last_prediction
    st.info(f"Mode: {mode} | Predicted: {label} | Confidence: {conf:.2f}")
else:
    st.write("Capture or upload an image to get prediction.")

st.subheader("Sentence Builder")
if sentence:
    display_sentence = " ".join(NUMBER_WORDS.get(token, token) for token in st.session_state.sentence)
    st.write(display_sentence)
else:
    st.write("Sentence is empty.")

word_map = {
    "H E L L O": "HELLO",
    "H I": "HI",
    "B Y E": "BYE",
}

if st.session_state.sentence and all(token in NUMBER_LABELS for token in st.session_state.sentence):
    st.success(f"Number sequence detected: {' '.join(st.session_state.sentence)}")

if sentence in word_map:
    st.success(f"Word Detected: {word_map[sentence]}")

act1, act2 = st.columns(2)
if act1.button("Speak Sentence"):
    if sentence:
        speak(" ".join(NUMBER_WORDS.get(token, token) for token in st.session_state.sentence))
    else:
        st.warning("Sentence is empty.")

if act2.button("Export dataset JSON"):
    st.download_button(
        "Download embeddings.json",
        data=json.dumps(db, indent=2),
        file_name="embeddings.json",
        mime="application/json",
        key="download_db",
    )

st.subheader("Training Dashboard")
sample_stats = {k: len(v) for k, v in db.items()}
if sample_stats:
    stats_df = pd.DataFrame(
        {"label": list(sample_stats.keys()), "samples": list(sample_stats.values())}
    ).sort_values("label")
    st.write("Per-label sample distribution")
    st.dataframe(stats_df, use_container_width=True)
    st.bar_chart(stats_df.set_index("label"))
else:
    st.info("No samples found yet. Save samples first to see dashboard metrics.")

eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    run_eval = st.button("Run holdout evaluation")
with eval_col2:
    st.caption("Uses stratified 80/20 split on current dataset for quick quality check.")

if run_eval:
    if clf is None or model_labels is None:
        st.warning("Train model first using train_lstm.py, then run evaluation.")
    else:
        eval_result, eval_error = evaluate_model_holdout(clf, db, model_labels)
        if eval_error:
            st.warning(eval_error)
        else:
            st.success(
                f"Holdout Accuracy: {eval_result['accuracy']:.3f} on {eval_result['test_samples']} test samples"
            )
            st.write("Confusion Matrix")
            st.dataframe(eval_result["cm_df"], use_container_width=True)
            st.write("Classification Report")
            st.dataframe(eval_result["report_df"], use_container_width=True)
            st.write("Misclassified Samples")
            if eval_result["mis_df"].empty:
                st.info("No misclassifications in this holdout split.")
            else:
                st.dataframe(eval_result["mis_df"], use_container_width=True)

st.markdown("### Advanced Features Included")
st.markdown("- Dual input modes: camera capture and image upload")
st.markdown("- No-dataset startup mode with heuristic gesture prediction")
st.markdown("- Few-shot learning by saving your own samples per label")
st.markdown("- Optional trained model inference (RandomForest)")
st.markdown("- Confidence-gated sentence builder and text-to-speech")
st.markdown("- Dedicated number-sign mode for digits 0-9")
st.markdown("- Real-time webcam stream with continuous detection")
st.markdown("- Confidence-trend smart lock for unstable frame suppression")
st.markdown("- Training dashboard with holdout accuracy and confusion matrix")
