import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)

SEVERITY_SCORES = {
    "Minor": 0.0,
    "Moderate": 1.0,
    "Severe": 2.0,
}

MARKET_COST_RANGES_USD = {
    "phone": {
        # Typical screen/backglass repair pricing is usually well below full replacement value.
        "Minor": (25.0, 85.0),
        "Moderate": (85.0, 190.0),
        "Severe": (190.0, 340.0),
    },
    "laptop": {
        "Minor": (60.0, 180.0),
        "Moderate": (180.0, 420.0),
        "Severe": (420.0, 820.0),
    },
}

# If a trained regressor exists, it can only influence the market estimate slightly.
MAX_MODEL_ADJUSTMENT_RATIO = 0.15


@dataclass
class CostEstimator:
    model: tf.keras.Model
    weights_loaded: bool

    def estimate(
        self,
        severity_label: str,
        device_type: str,
        severity_confidence: float,
        severity_probabilities: Dict[str, float],
    ) -> Tuple[float, str]:
        if severity_label not in SEVERITY_SCORES:
            raise ValueError(f"Unknown severity: {severity_label}")

        if device_type not in MARKET_COST_RANGES_USD:
            raise ValueError(f"Unknown device type: {device_type}")

        ranges = MARKET_COST_RANGES_USD[device_type]

        # Weighted expected value from classifier probabilities produces smoother, more realistic estimates.
        weighted_expected = 0.0
        probability_mass = 0.0
        for label, (low, high) in ranges.items():
            prob = float(severity_probabilities.get(label, 0.0))
            midpoint = (low + high) / 2.0
            weighted_expected += prob * midpoint
            probability_mass += prob

        if probability_mass <= 0:
            selected_low, selected_high = ranges[severity_label]
            weighted_expected = (selected_low + selected_high) / 2.0

        selected_low, selected_high = ranges[severity_label]

        # Move within selected severity range based on confidence while keeping estimate bounded.
        confidence = float(np.clip(severity_confidence, 0.0, 1.0))
        # A cracked screen usually prices closer to the low-to-mid part of the band unless confidence is very high.
        within_band = selected_low + (0.15 + 0.5 * confidence) * (selected_high - selected_low)

        # Keep the estimate anchored to the class probabilities while avoiding replacement-like pricing.
        blended = 0.72 * weighted_expected + 0.28 * within_band
        estimate = float(np.clip(blended, selected_low, selected_high))

        note = f"Market-range estimate for {device_type} using severity probabilities"

        if self.weights_loaded:
            score = np.array([[SEVERITY_SCORES[severity_label]]], dtype=np.float32)
            model_raw = float(self.model.predict(score, verbose=0)[0][0])
            normalized = float(np.clip(model_raw / 2.0, 0.0, 1.0))
            model_estimate = selected_low + normalized * (selected_high - selected_low)
            delta = model_estimate - estimate
            # Bound the learned adjustment so it can refine but not dominate the heuristic estimate.
            estimate = estimate + float(np.clip(delta, -MAX_MODEL_ADJUSTMENT_RATIO * estimate, MAX_MODEL_ADJUSTMENT_RATIO * estimate))
            note = f"Market-range estimate for {device_type} with bounded regressor adjustment"

        return round(estimate, 2), note


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
