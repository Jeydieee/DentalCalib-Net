import cv2
import numpy as np
from pathlib import Path


def laplacian_variance(img):
    """Sharpness/blur measure. Lower value = blurrier image."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def pixel_std_dev(img):
    """Exposure/contrast measure."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(gray.std())


def jpeg_block_artifact_score(img, block_size=8):
    """Compression artifact measure via mean absolute difference across 8x8 block boundaries."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float64)
    h, w = gray.shape

    diffs = []
    for x in range(block_size, w, block_size):
        if x >= w:
            break
        left = gray[:, x - 1]
        right = gray[:, x]
        diffs.append(np.mean(np.abs(left - right)))

    for y in range(block_size, h, block_size):
        if y >= h:
            break
        top = gray[y - 1, :]
        bottom = gray[y, :]
        diffs.append(np.mean(np.abs(top - bottom)))

    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def extract_quality_features(img_path):
    """Loads an image and returns all three features as a dict."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    return {
        'laplacian_variance': laplacian_variance(img),
        'pixel_std_dev': pixel_std_dev(img),
        'jpeg_block_artifact_score': jpeg_block_artifact_score(img),
    }


if __name__ == "__main__":
    clean_path = r"D:/DentalCalib-Net/dataset_split/test/test_0.png"
    blur_s5_path = r"D:/DentalCalib-Net/dataset_split/ood/motion_blur/S5/test_0.png"
    noise_s5_path = r"D:/DentalCalib-Net/dataset_split/ood/gaussian_noise/S5/test_0.png"
    jpeg_s5_path = r"D:/DentalCalib-Net/dataset_split/ood/jpeg_compression/S5/test_0.png"

    print("Clean:", extract_quality_features(clean_path))
    print("Motion blur S5:", extract_quality_features(blur_s5_path))
    print("Gaussian noise S5:", extract_quality_features(noise_s5_path))
    print("JPEG compression S5:", extract_quality_features(jpeg_s5_path))