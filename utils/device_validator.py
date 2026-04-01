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
        "ipod",
        "hand-held_computer",
    },
    "laptop": {
        "laptop",
        "notebook",
        "desktop_computer",
        "monitor",
        "screen",
    },
}

# MobileNet top-5 confidence can be low on damaged-device photos, so keep threshold permissive.
MIN_ACCEPT_SCORE = 0.03
# Reject only when the model is confidently identifying a non-device object.
STRONG_NON_DEVICE_SCORE = 0.35


def _has_device_hint(label: str, expected_device: str) -> bool:
    normalized = label.lower().replace("_", "-")
    if expected_device == "phone":
        return any(token in normalized for token in ["phone", "telephone", "cell", "hand-held", "ipod"])
    if expected_device == "laptop":
        return any(token in normalized for token in ["laptop", "notebook", "computer", "screen", "monitor"])
    return False


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

    allowed = {label.lower() for label in ALLOWED_LABELS[expected_device]}
    for label, score in simplified:
        normalized_label = label.lower()
        if normalized_label in allowed and score >= MIN_ACCEPT_SCORE:
            return True, "ok", simplified

    # If model is uncertain, do not hard-fail valid photos (common on damaged close-up images).
    for label, score in simplified:
        if _has_device_hint(label, expected_device) and score >= 0.01:
            return True, "ok", simplified

    top_label, top_score = simplified[0]
    if top_score < STRONG_NON_DEVICE_SCORE:
        return True, "ok", simplified

    readable_top = ", ".join([f"{label} ({score:.2f})" for label, score in simplified[:3]])
    reason = (
        f"Uploaded image does not look like a {expected_device}. "
        f"Top detections: {readable_top}. Please upload a clear {expected_device} photo only."
    )
    return False, reason, simplified
