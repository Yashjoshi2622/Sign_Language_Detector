import json
import math
import os
from typing import Dict, List

import numpy as np


OUT_PATH = "data/embeddings.json"
SAMPLES_PER_LABEL = 180
SEED = 42

# MediaPipe hand landmark indices
# wrist=0
# thumb: 1,2,3,4
# index: 5,6,7,8
# middle: 9,10,11,12
# ring: 13,14,15,16
# pinky: 17,18,19,20


FINGER_IDX = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# For each digit: (thumb, index, middle, ring, pinky) extended flags.
# 6-9 are synthetic conventions for bootstrapping only.
DIGIT_PATTERNS = {
    "0": (0, 0, 0, 0, 0),
    "1": (0, 1, 0, 0, 0),
    "2": (0, 1, 1, 0, 0),
    "3": (1, 1, 1, 0, 0),
    "4": (0, 1, 1, 1, 1),
    "5": (1, 1, 1, 1, 1),
    "6": (1, 0, 0, 0, 1),
    "7": (1, 0, 0, 1, 0),
    "8": (1, 0, 1, 0, 0),
    "9": (1, 1, 0, 0, 0),
}


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points - points[0]
    scale = np.max(np.linalg.norm(points, axis=1)) + 1e-6
    return points / scale


def rotate_points(points: np.ndarray, degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    rot = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=np.float32)
    return points @ rot.T


def base_hand() -> np.ndarray:
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = [0.0, 0.0]

    # MCP anchors (index->pinky)
    points[5] = [-0.28, -0.08]
    points[9] = [-0.10, -0.10]
    points[13] = [0.10, -0.10]
    points[17] = [0.28, -0.08]

    # Thumb base joints
    points[1] = [-0.16, 0.02]
    points[2] = [-0.28, 0.02]
    points[3] = [-0.36, 0.03]
    points[4] = [-0.44, 0.04]
    return points


def set_finger(points: np.ndarray, finger: str, extended: int) -> None:
    a, b, c, d = FINGER_IDX[finger]

    if finger == "thumb":
        if extended:
            points[a] = [-0.16, 0.01]
            points[b] = [-0.30, -0.02]
            points[c] = [-0.42, -0.05]
            points[d] = [-0.55, -0.08]
        else:
            points[a] = [-0.14, 0.03]
            points[b] = [-0.20, 0.06]
            points[c] = [-0.18, 0.10]
            points[d] = [-0.12, 0.13]
        return

    mcp = points[a]
    if extended:
        points[b] = mcp + np.array([0.00, -0.20], dtype=np.float32)
        points[c] = mcp + np.array([0.00, -0.40], dtype=np.float32)
        points[d] = mcp + np.array([0.00, -0.62], dtype=np.float32)
    else:
        points[b] = mcp + np.array([0.03, -0.06], dtype=np.float32)
        points[c] = mcp + np.array([0.05, -0.01], dtype=np.float32)
        points[d] = mcp + np.array([0.07, 0.03], dtype=np.float32)


def generate_embedding(pattern: tuple[int, int, int, int, int], rng: np.random.Generator) -> np.ndarray:
    points = base_hand()
    thumb, index, middle, ring, pinky = pattern

    set_finger(points, "thumb", thumb)
    set_finger(points, "index", index)
    set_finger(points, "middle", middle)
    set_finger(points, "ring", ring)
    set_finger(points, "pinky", pinky)

    # Pose augmentation
    points = rotate_points(points, float(rng.uniform(-18, 18)))
    points *= float(rng.uniform(0.9, 1.1))
    points += rng.normal(0.0, 0.02, size=points.shape).astype(np.float32)

    points = normalize_points(points)
    return points.flatten().astype(np.float32)


def build_dataset(samples_per_label: int = SAMPLES_PER_LABEL) -> Dict[str, List[List[float]]]:
    rng = np.random.default_rng(SEED)
    dataset: Dict[str, List[List[float]]] = {}

    for label, pattern in DIGIT_PATTERNS.items():
        samples = [generate_embedding(pattern, rng).tolist() for _ in range(samples_per_label)]
        dataset[label] = samples

    return dataset


def main() -> None:
    os.makedirs("data", exist_ok=True)
    dataset = build_dataset()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(f"Saved synthetic dataset to {OUT_PATH}")
    print("Labels:", ", ".join(sorted(dataset.keys())))
    print("Samples per label:", len(dataset["0"]))


if __name__ == "__main__":
    main()
