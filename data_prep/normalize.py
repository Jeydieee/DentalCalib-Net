import cv2
import numpy as np
from pathlib import Path


def load_and_normalize(img_path):
    """Load an image and apply min-max normalization to [0, 1].
    Returns a float32 array. Use this at model input time (Stage 2 onward),
    not for saving intermediate files -- PNG can't store normalized floats.
    """
    img = cv2.imread(str(img_path))  # already resized + grayscale from Stage 1
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    img = img.astype(np.float32)

    img_min = img.min()
    img_max = img.max()

    if img_max - img_min < 1e-8:
        # degenerate case: flat image (all same pixel value), avoid divide-by-zero
        return np.zeros_like(img)

    normalized = (img - img_min) / (img_max - img_min)
    return normalized


def verify_normalization(img_path):
    """Quick sanity check: confirms normalization produces the expected
    [0, 1] range and preserves image shape. Not a full pipeline step --
    just a spot-check to run against a real image before trusting the function."""
    normalized = load_and_normalize(img_path)

    print(f"Image: {img_path}")
    print(f"shape: {normalized.shape}")
    print(f"dtype: {normalized.dtype}")
    print(f"min: {normalized.min():.6f}")
    print(f"max: {normalized.max():.6f}")

    assert 0.0 <= normalized.min() <= 1e-6, "min should be ~0"
    assert 0.999 <= normalized.max() <= 1.0, "max should be ~1"
    print("OK - normalization verified")


if __name__ == "__main__":
    # spot-check against one real resized image from your dataset_split folder
    verify_normalization(r"D:/DentalCalib-Net/dataset_split/train/train_5.png")