import os
from typing import List

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")


def list_sample_images() -> List[str]:
    if not os.path.isdir(SAMPLE_DIR):
        return []

    images = []
    for name in os.listdir(SAMPLE_DIR):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(SAMPLE_DIR, name))

    return sorted(images)
