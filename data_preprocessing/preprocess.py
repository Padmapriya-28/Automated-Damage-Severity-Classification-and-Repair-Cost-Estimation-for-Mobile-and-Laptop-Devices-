from typing import Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

MODEL_INPUTS = {
    "resnet50": (224, 224, resnet50.preprocess_input),
    "mobilenetv2": (224, 224, mobilenet_v2.preprocess_input),
    "efficientnetb0": (224, 224, efficientnet.preprocess_input),
}


def preprocess_image(image: Image.Image, model_name: str) -> Tuple[np.ndarray, Image.Image]:
    if model_name not in MODEL_INPUTS:
        raise ValueError(f"Unsupported model: {model_name}")

    size, _, preprocess_fn = MODEL_INPUTS[model_name]
    resized = image.resize((size, size))
    array = np.array(resized, dtype=np.float32)
    array = preprocess_fn(array)
    array = np.expand_dims(array, axis=0)

    return array, resized
