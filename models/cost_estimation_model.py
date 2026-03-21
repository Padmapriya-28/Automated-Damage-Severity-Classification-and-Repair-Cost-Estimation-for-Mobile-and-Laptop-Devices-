import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)

SEVERITY_SCORES = {
    "Minor": 0.0,
    "Moderate": 1.0,
    "Severe": 2.0,
}

DEVICE_FACTORS = {
    "phone": 1.0,
    "laptop": 1.65,
}

HEURISTIC_COST = {
    "phone": {"base": 90.0, "multiplier": 180.0},
    "laptop": {"base": 190.0, "multiplier": 340.0},
}


@dataclass
class CostEstimator:
    model: tf.keras.Model
    weights_loaded: bool

    def estimate(self, severity_label: str, device_type: str) -> Tuple[float, str]:
        if severity_label not in SEVERITY_SCORES:
            raise ValueError(f"Unknown severity: {severity_label}")

        if device_type not in DEVICE_FACTORS:
            raise ValueError(f"Unknown device type: {device_type}")

        score = np.array([[SEVERITY_SCORES[severity_label]]], dtype=np.float32)
        device_factor = DEVICE_FACTORS[device_type]

        if self.weights_loaded:
            prediction = float(self.model.predict(score, verbose=0)[0][0])
            adjusted = max(prediction, 0.0) * device_factor
            return round(adjusted, 2), f"Model-based estimate ({device_type})"

        base = HEURISTIC_COST[device_type]["base"]
        multiplier = HEURISTIC_COST[device_type]["multiplier"]
        estimate = base + multiplier * SEVERITY_SCORES[severity_label]
        return round(estimate, 2), f"Heuristic estimate for {device_type} (replace with trained regressor)"


def build_cost_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(1,), name="severity_score")
    x = layers.Dense(8, activation="relu")(inputs)
    x = layers.Dense(4, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear", name="cost")(x)
    model = models.Model(inputs, outputs, name="repair_cost_regressor")
    model.compile(optimizer="adam", loss="mse")
    return model


def load_cost_weights(model: tf.keras.Model) -> bool:
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    weights_path = os.path.join(weights_dir, "cost_regressor.keras")
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        logger.info("Loaded cost estimator weights from %s", weights_path)
        return True
    return False


@lru_cache(maxsize=1)
def get_cost_estimator() -> CostEstimator:
    model = build_cost_model()
    loaded = load_cost_weights(model)
    if not loaded:
        logger.info("Cost estimator running with heuristic fallback")
    return CostEstimator(model=model, weights_loaded=loaded)
