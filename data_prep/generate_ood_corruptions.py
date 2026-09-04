import cv2
import numpy as np
from pathlib import Path
import imgaug.augmenters as iaa

TEST_DIR = r"D:/DentalCalib-Net/dataset_split/test"
OUTPUT_BASE = r"D:/DentalCalib-Net/dataset_split/ood"

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# Approximated from Hendrycks & Dietterich (2019) ImageNet-C constants,
# converted into imgaug's parameter space. These are NOT identical to
# H&D's original algorithm outputs -- imgaug uses different internal math
# for each augmenter -- but are calibrated to track the same severity
# progression as closely as imgaug's API allows.

# H&D gaussian_noise c = [.08, .12, .18, .26, .38] (fraction of [0,1] range)
# -> converted to 0-255 scale for imgaug's AdditiveGaussianNoise
GAUSSIAN_NOISE_SCALE = [.08 * 255, .12 * 255, .18 * 255, .26 * 255, .38 * 255]

# H&D motion_blur c = [(10,3), (15,5), (15,8), (15,12), (20,15)] -- (radius, sigma)
# -> imgaug's MotionBlur takes kernel size k; approximate k as 2*radius+1
MOTION_BLUR_K = [21, 31, 31, 31, 41]

# H&D brightness c = [.1, .2, .3, .4, .5] (added to HSV V-channel, [0,1] scale)
# -> imgaug's Multiply scales pixel values; approximate as a multiplicative
# brightening factor roughly equivalent in visual effect
BRIGHTNESS_MULT = [1.10, 1.20, 1.30, 1.40, 1.50]

# H&D jpeg_compression c = [25, 18, 15, 10, 7] -- JPEG quality
# -> imgaug's JpegCompression takes compression amount (inverse of quality)
JPEG_COMPRESSION = [100 - 25, 100 - 18, 100 - 15, 100 - 10, 100 - 7]


def get_augmenter(corruption_name, severity):
    idx = severity - 1

    if corruption_name == 'gaussian_noise':
        return iaa.AdditiveGaussianNoise(scale=GAUSSIAN_NOISE_SCALE[idx])

    elif corruption_name == 'motion_blur':
        return iaa.MotionBlur(k=MOTION_BLUR_K[idx], angle=(-45, 45))

    elif corruption_name == 'brightness_variation':
        return iaa.Multiply(BRIGHTNESS_MULT[idx])

    elif corruption_name == 'jpeg_compression':
        return iaa.JpegCompression(compression=JPEG_COMPRESSION[idx])

    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")


CORRUPTION_NAMES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']


def generate_ood_sets(test_dir=TEST_DIR, output_base=OUTPUT_BASE, seed=42):
    np.random.seed(seed)

    test_dir = Path(test_dir)
    image_files = sorted(test_dir.glob('*.png'))

    print(f"Found {len(image_files)} test images.\n")

    for corruption_name in CORRUPTION_NAMES:
        for severity in SEVERITY_LEVELS:
            out_dir = Path(output_base) / corruption_name / f"S{severity}"
            out_dir.mkdir(parents=True, exist_ok=True)

            augmenter = get_augmenter(corruption_name, severity)
            processed = 0

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                corrupted = augmenter(image=img)

                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), corrupted)
                processed += 1

            print(f"{corruption_name} S{severity}: {processed} images -> {out_dir}")

    print("\nDone. 20 OOD sets generated.")


if __name__ == "__main__":
    generate_ood_sets()