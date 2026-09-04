import shutil
from pathlib import Path

SOURCE_DIRS = {
    'train': r"D:/DentalCalib-Net/dataset_split/train",
    'val': r"D:/DentalCalib-Net/dataset_split/val",
}

OUTPUT_BASE = r"D:\DentalCalib-Net\yolo_dataset"

def copy_images(partition, source_dir, labels_dir, images_out_dir):
    """Copy only images that have a corresponding label file --
    keeps images and labels in sync, one-to-one."""
    source_dir = Path(source_dir)
    labels_dir = Path(labels_dir)
    images_out_dir = Path(images_out_dir)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    label_files = {f.stem for f in labels_dir.glob('*.txt')}

    copied, missing = 0, []
    for label_stem in label_files:
        src_img = source_dir / f"{label_stem}.png"
        if not src_img.exists():
            missing.append(label_stem)
            continue
        shutil.copy2(src_img, images_out_dir / f"{label_stem}.png")
        copied += 1

    print(f"{partition}: copied {copied} images, {len(missing)} missing")
    if missing:
        print(f"  missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    copy_images('train', SOURCE_DIRS['train'],
        f"{OUTPUT_BASE}/labels/train", f"{OUTPUT_BASE}/images/train")
    copy_images('val', SOURCE_DIRS['val'],
        f"{OUTPUT_BASE}/labels/val", f"{OUTPUT_BASE}/images/val")