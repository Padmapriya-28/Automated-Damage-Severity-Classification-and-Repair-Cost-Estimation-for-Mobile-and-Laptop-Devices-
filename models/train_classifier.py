import random
import sys
from pathlib import Path
from typing import List, Tuple

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
EPOCHS = 4
MODEL_NAME = "efficientnetb0"


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


def build_tf_dataset(paths: List[str], labels: List[int], training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        image = efficientnet.preprocess_input(image)
        image.set_shape((224, 224, 3))
        one_hot = tf.one_hot(label, depth=len(LABELS))
        return image, one_hot

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(256, seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def train() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    root = Path(__file__).resolve().parents[1] / "data"
    paths, labels = collect_dataset(root)

    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = zip(*combined)

    split_idx = max(1, int(0.8 * len(paths)))
    train_paths, val_paths = list(paths[:split_idx]), list(paths[split_idx:])
    train_labels, val_labels = list(labels[:split_idx]), list(labels[split_idx:])

    if not val_paths:
        val_paths = train_paths[: max(1, len(train_paths) // 5)]
        val_labels = train_labels[: len(val_paths)]

    train_ds = build_tf_dataset(train_paths, train_labels, training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, training=False)

    model = build_model(MODEL_NAME)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / f"{MODEL_NAME}.weights.h5"
    model.save_weights(str(out_path))
    print(f"Saved trained weights to: {out_path}")


if __name__ == "__main__":
    train()
