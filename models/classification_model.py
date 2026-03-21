import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

logger = logging.getLogger(__name__)

LABELS = ["Minor", "Moderate", "Severe"]
LAST_CONV_LAYER = {
    "resnet50": "conv5_block3_out",
    "mobilenetv2": "Conv_1",
    "efficientnetb0": "top_conv",
}


@dataclass
class DamageClassifier:
    model_name: str
    model: tf.keras.Model
    last_conv_layer: str
    label_to_index: Dict[str, int]

    def predict(self, image_array: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        logits = self.model.predict(image_array, verbose=0)[0]
        probs = tf.nn.softmax(logits).numpy().tolist()
        best_idx = int(np.argmax(probs))
        label = LABELS[best_idx]
        confidence = float(probs[best_idx])
        prob_map = {LABELS[i]: float(prob) for i, prob in enumerate(probs)}
        return label, confidence, prob_map


def build_model(model_name: str) -> tf.keras.Model:
    if model_name == "resnet50":
        base = resnet50.ResNet50(weights="imagenet", include_top=False, pooling="avg")
    elif model_name == "mobilenetv2":
        base = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    elif model_name == "efficientnetb0":
        base = efficientnet.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    features = base(inputs, training=False)
    outputs = layers.Dense(
        len(LABELS),
        activation=None,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="damage_logits",
    )(features)
    model = models.Model(inputs, outputs, name=f"damage_classifier_{model_name}")
    return model


def load_weights_if_available(model: tf.keras.Model, model_name: str) -> str:
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    candidate_paths = [
        os.path.join(weights_dir, f"{model_name}.weights.h5"),
        os.path.join(weights_dir, f"{model_name}.keras"),
    ]
    for weights_path in candidate_paths:
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
            return f"Loaded weights from {weights_path}"
    return "Using base ImageNet backbone with untrained classification head"


@lru_cache(maxsize=1)
def get_classifier(model_name: str = "efficientnetb0") -> DamageClassifier:
    model = build_model(model_name)
    note = load_weights_if_available(model, model_name)
    logger.info("Damage classifier ready: %s", note)

    label_map = {label: idx for idx, label in enumerate(LABELS)}
    return DamageClassifier(
        model_name=model_name,
        model=model,
        last_conv_layer=LAST_CONV_LAYER[model_name],
        label_to_index=label_map,
    )
