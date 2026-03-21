from functools import lru_cache
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

# ImageNet labels used by MobileNetV2 for strict device validation.
ALLOWED_LABELS = {
    "phone": {
        "cellular_telephone",
        "pay-phone",
        "dial_telephone",
    },
    "laptop": {
        "laptop",
        "notebook",
        "desktop_computer",
        "monitor",
        "screen",
    },
}


def _prepare(image: Image.Image) -> np.ndarray:
    resized = image.resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Input image must be RGB")
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


@lru_cache(maxsize=1)
def _get_model() -> tf.keras.Model:
    return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)


def validate_device_image(image: Image.Image, expected_device: str) -> Tuple[bool, str, List[Tuple[str, float]]]:
    if expected_device not in ALLOWED_LABELS:
        return False, f"Unsupported device type: {expected_device}", []

    model = _get_model()
    preds = model.predict(_prepare(image), verbose=0)
    top = decode_predictions(preds, top=5)[0]
    simplified = [(label, float(score)) for _, label, score in top]

    allowed = ALLOWED_LABELS[expected_device]
    for label, score in simplified:
        if label in allowed and score >= 0.2:
            return True, "ok", simplified

    readable_top = ", ".join([f"{label} ({score:.2f})" for label, score in simplified[:3]])
    reason = (
        f"Uploaded image does not look like a {expected_device}. "
        f"Top detections: {readable_top}. Please upload a clear {expected_device} photo only."
    )
    return False, reason, simplified
