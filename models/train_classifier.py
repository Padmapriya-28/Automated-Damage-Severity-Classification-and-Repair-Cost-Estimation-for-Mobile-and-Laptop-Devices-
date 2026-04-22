import random
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification_model import LABELS, build_model

RANDOM_SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
VAL_SPLIT = 0.2
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 0
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = 180
MODEL_NAME = "efficientnetb0"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_classifier(model: tf.keras.Model, val_ds: tf.data.Dataset) -> Dict:
    logits = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    y_true_chunks = []
    for _, batch_labels in val_ds:
        y_true_chunks.append(np.argmax(batch_labels.numpy(), axis=1))
    y_true = np.concatenate(y_true_chunks, axis=0)

    num_classes = len(LABELS)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1

    per_class = []
    precision_values = []
    recall_values = []
    f1_values = []

    for idx, label in enumerate(LABELS):
        tp = int(confusion[idx, idx])
        fp = int(np.sum(confusion[:, idx]) - tp)
        fn = int(np.sum(confusion[idx, :]) - tp)
        support = int(np.sum(confusion[idx, :]))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        class_accuracy = _safe_div(tp, support)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_class.append(
            {
                "label": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(class_accuracy, 4),
                "support": support,
            }
        )

    overall_accuracy = _safe_div(float(np.sum(np.diag(confusion))), float(np.sum(confusion)))
    return {
        "summary": {
            "accuracy": round(overall_accuracy, 4),
            "macro_precision": round(float(np.mean(precision_values)), 4),
            "macro_recall": round(float(np.mean(recall_values)), 4),
            "macro_f1": round(float(np.mean(f1_values)), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion.astype(int).tolist(),
        },
    }


def collect_dataset(root: Path) -> Tuple[List[str], List[int]]:
    broken_dir = root / "Image_brokenphones"
    normal_dir = root / "Image_phones"

    if not broken_dir.exists() or not normal_dir.exists():
        raise FileNotFoundError("Expected data/Image_brokenphones and data/Image_phones directories")

    paths: List[str] = []
    labels: List[int] = []

    for p in sorted(broken_dir.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(2)  # Severe

    for p in sorted(normal_dir.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(0)  # Minor

    if not paths:
        raise ValueError("No images found for training")

    return paths, labels


def stratified_split(paths: List[str], labels: List[int], val_ratio: float) -> Tuple[List[str], List[int], List[str], List[int]]:
    by_label: Dict[int, List[str]] = {}
    for path, label in zip(paths, labels):
        by_label.setdefault(label, []).append(path)

    train_paths: List[str] = []
    train_labels: List[int] = []
    val_paths: List[str] = []
    val_labels: List[int] = []

    for label, class_paths in by_label.items():
        random.shuffle(class_paths)
        if len(class_paths) == 1:
            train_chunk = class_paths
            val_chunk: List[str] = []
        else:
            val_count = max(1, int(round(len(class_paths) * val_ratio)))
            val_count = min(val_count, len(class_paths) - 1)
            val_chunk = class_paths[:val_count]
            train_chunk = class_paths[val_count:]

        train_paths.extend(train_chunk)
        train_labels.extend([label] * len(train_chunk))
        val_paths.extend(val_chunk)
        val_labels.extend([label] * len(val_chunk))

    if not val_paths:
        val_size = max(1, int(len(train_paths) * val_ratio))
        val_paths = train_paths[:val_size]
        val_labels = train_labels[:val_size]
        train_paths = train_paths[val_size:]
        train_labels = train_labels[val_size:]

    combined_train = list(zip(train_paths, train_labels))
    random.shuffle(combined_train)
    train_paths, train_labels = zip(*combined_train)

    combined_val = list(zip(val_paths, val_labels))
    random.shuffle(combined_val)
    val_paths, val_labels = zip(*combined_val)

    return list(train_paths), list(train_labels), list(val_paths), list(val_labels)


def build_tf_dataset(paths: List[str], labels: List[int], training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="train_augmentation",
    )

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        if training:
            image = augmenter(image, training=True)
        image = efficientnet.preprocess_input(image)
        image.set_shape((224, 224, 3))
        one_hot = tf.one_hot(label, depth=len(LABELS))
        return image, one_hot

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(256, seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def class_distribution(labels: List[int]) -> Dict[str, int]:
    counts = {label_name: 0 for label_name in LABELS}
    for idx in labels:
        counts[LABELS[idx]] += 1
    return {k: v for k, v in counts.items() if v > 0}


def _get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find EfficientNet backbone layer for fine-tuning")


def _merge_history(*histories: tf.keras.callbacks.History) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend([float(v) for v in values])
    return merged


def train() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    root = Path(__file__).resolve().parents[1] / "data"
    paths, labels = collect_dataset(root)
    train_paths, train_labels, val_paths, val_labels = stratified_split(paths, labels, VAL_SPLIT)

    train_ds = build_tf_dataset(train_paths, train_labels, training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, training=False)

    model = build_model(MODEL_NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=head_callbacks,
        verbose=1,
    )

    backbone = _get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    merged_history = _merge_history(head_history)
    if FINE_TUNE_EPOCHS > 0:
        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
        ]

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=fine_tune_callbacks,
            verbose=1,
        )
        merged_history = _merge_history(head_history, fine_tune_history)

    metrics = evaluate_classifier(model, val_ds)
    metrics_payload = {
        "model_name": MODEL_NAME,
        "evaluated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": {
            "train_samples": len(train_paths),
            "validation_samples": len(val_paths),
            "class_distribution_train": class_distribution(train_labels),
            "class_distribution_validation": class_distribution(val_labels),
        },
        "summary": metrics["summary"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "training_history": {
            "loss": [round(float(v), 4) for v in merged_history.get("loss", [])],
            "accuracy": [round(float(v), 4) for v in merged_history.get("accuracy", [])],
            "val_loss": [round(float(v), 4) for v in merged_history.get("val_loss", [])],
            "val_accuracy": [round(float(v), 4) for v in merged_history.get("val_accuracy", [])],
        },
    }

    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / f"{MODEL_NAME}.weights.h5"
    model.save_weights(str(out_path))
    metrics_path = weights_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved trained weights to: {out_path}")
    print(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    train()
