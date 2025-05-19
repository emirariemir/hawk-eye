from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import json
import os
from tqdm import tqdm

def evaluate_frcnn(model, val_dataset, annotation_path, device, output_json="frcnn_predictions.json"):
    model.eval()
    model.to(device)

    coco_gt = COCO(annotation_path)
    coco_results = []

    img_ids = []

    for img, target in tqdm(val_dataset, desc="Running inference"):
        img = img.to(device)
        with torch.no_grad():
            outputs = model([img])[0]

        image_id = int(target['image_id'].item())
        img_ids.append(image_id)

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x_min, y_min, width, height],
                "score": float(score)
            })

    # Save predictions in COCO JSON format
    with open(output_json, "w") as f:
        json.dump(coco_results, f, indent=4)

    # Load predictions and run evaluation
    coco_dt = coco_gt.loadRes(output_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
