import os
import json
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Prepare output label folder
    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build category ID to index map
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    class_names = [cat['name'] for cat in coco['categories']]

    # Build image ID to file name and image size map
    image_id_map = {img['id']: img for img in coco['images']}

    # Create empty label files for each image
    for image in coco['images']:
        label_path = labels_dir / Path(image['file_name']).with_suffix('.txt')
        label_path.write_text("")  # Empty initially

    # Fill label files with YOLO annotations
    for ann in coco['annotations']:
        img_info = image_id_map[ann['image_id']]
        img_w, img_h = img_info['width'], img_info['height']
        category_index = cat_id_to_index[ann['category_id']]

        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w /= img_w
        h /= img_h

        yolo_line = f"{category_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        label_path = labels_dir / Path(img_info['file_name']).with_suffix('.txt')
        with open(label_path, 'a') as f:
            f.write(yolo_line)

    print(f"✅ COCO annotations converted to YOLO format in: {labels_dir}")
    return class_names


def generate_data_yaml(train_img_dir, val_img_dir, class_names, output_path="data.yaml"):
    data_yaml = {
        "train": str(Path(train_img_dir).resolve()),
        "val": str(Path(val_img_dir).resolve()),
        "nc": len(class_names),
        "names": class_names
    }

    import yaml
    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"✅ data.yaml created at: {output_path}")


# ==== 🔁 Run this block ====
if __name__ == "__main__":
    # Edit these paths according to your structure
    dataset_root = "dataset"

    train_folder = "train_teene_reduced"
    val_folder = "validation_teene_reduced"

    # Convert annotations to YOLO labels
    train_classes = convert_coco_to_yolo(
        coco_json_path=f"{dataset_root}/{train_folder}/annotations.json",
        images_dir=f"{dataset_root}/{train_folder}/images",
        output_dir=f"{dataset_root}/{train_folder}"
    )

    val_classes = convert_coco_to_yolo(
        coco_json_path=f"{dataset_root}/{val_folder}/annotations.json",
        images_dir=f"{dataset_root}/{val_folder}/images",
        output_dir=f"{dataset_root}/{val_folder}"
    )

    # Make sure train/val classes match
    assert train_classes == val_classes, "Mismatch in class names between train and val!"

    # Create data.yaml
    generate_data_yaml(
        train_img_dir=f"{dataset_root}/{train_folder}/images",
        val_img_dir=f"{dataset_root}/{val_folder}/images",
        class_names=train_classes,
        output_path="data.yaml"
    )
