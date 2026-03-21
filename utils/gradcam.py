import numpy as np
import tensorflow as tf
from PIL import Image


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(heatmap, 0, 1)
    red = heatmap
    green = np.sqrt(heatmap)
    blue = np.zeros_like(heatmap)
    return np.stack([red, green, blue], axis=-1)


def generate_gradcam_overlay(
    model: tf.keras.Model,
    image_array: np.ndarray,
    display_image: Image.Image,
    layer_name: str,
    class_index: int,
) -> Image.Image:
    conv_layer = _resolve_conv_layer(model, layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    heatmap = heatmap.numpy()

    heatmap_rgb = _apply_colormap(heatmap)
    heatmap_rgb = Image.fromarray(np.uint8(heatmap_rgb * 255))
    heatmap_rgb = heatmap_rgb.resize(display_image.size)

    overlay = Image.blend(display_image, heatmap_rgb, alpha=0.45)
    return overlay


def _resolve_conv_layer(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer:
    try:
        return model.get_layer(layer_name)
    except Exception:
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(layer_name)
            except Exception:
                continue

    raise ValueError(f"Could not find Grad-CAM layer '{layer_name}' in model or nested submodels")
