import json
import os

# === Paths ===
original_ann_path = 'archive/train/annotations.json'
reduced_img_dir = 'archive/train_10k/images'
output_ann_path = 'archive/train_10k/annotations.json'

print("📦 Loading original annotations...")
with open(original_ann_path, 'r') as f:
    coco = json.load(f)

# === Get valid image filenames ===
valid_filenames = set(os.listdir(reduced_img_dir))

# === Filter images ===
filtered_images = [img for img in coco['images'] if img['file_name'] in valid_filenames]
valid_image_ids = set(img['id'] for img in filtered_images)

# === Filter annotations ===
filtered_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in valid_image_ids]

# === Create reduced COCO dataset ===
filtered_coco = {
    'info': coco.get('info', {}),
    'licenses': coco.get('licenses', []),
    'categories': coco.get('categories', []),
    'images': filtered_images,
    'annotations': filtered_annotations
}

# === Save new annotations ===
with open(output_ann_path, 'w') as f:
    json.dump(filtered_coco, f)

print(f"✅ Saved reduced annotations to {output_ann_path}")
