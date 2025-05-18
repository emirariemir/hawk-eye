import os
import json
import shutil
import random
from pathlib import Path

def reduce_dataset(split_name: str, root_dir: str = "dataset", max_images: int = 3000):
    """
    Reduce the dataset split (train or validation) to `max_images` entries.
    Only includes entries that have a valid image file.
    """
    split_dir = Path(root_dir) / split_name
    images_dir = split_dir / "images"
    annotations_path = split_dir / "annotations.json"

    reduced_dir = Path(root_dir) / f"{split_name}_teene_reduced"
    reduced_images_dir = reduced_dir / "images"
    reduced_annotations_path = reduced_dir / "annotations.json"

    # Create directories
    reduced_images_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(annotations_path, "r") as f:
        data = json.load(f)

    all_images = data["images"]
    all_annotations = data["annotations"]
    categories = data["categories"]

    kept_images = []
    kept_image_ids = set()
    count = 0
    for img in all_images:
        src_img_path = images_dir / img["file_name"]
        if src_img_path.exists():
            dst_img_path = reduced_images_dir / img["file_name"]
            shutil.copy(src_img_path, dst_img_path)
            kept_images.append(img)
            kept_image_ids.add(img["id"])
            count += 1
            if count >= max_images:
                break
        else:
            print(f"⚠️ Missing file, skipping: {img['file_name']}")

    # Filter annotations to only include the ones for valid images
    reduced_annotations = [ann for ann in all_annotations if ann["image_id"] in kept_image_ids]

    # Write reduced annotations
    reduced_data = {
        "images": kept_images,
        "annotations": reduced_annotations,
        "categories": categories
    }

    with open(reduced_annotations_path, "w") as f:
        json.dump(reduced_data, f, indent=4)

    print(f"✅ {split_name} dataset reduced to {len(kept_images)} images and saved in '{reduced_dir}'")

if __name__ == "__main__":
    reduce_dataset("train", root_dir="dataset", max_images=300)
    reduce_dataset("validation", root_dir="dataset", max_images=300)
