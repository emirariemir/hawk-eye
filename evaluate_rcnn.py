from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import json
import os
from tqdm import tqdm

def compute_validation_loss(model, val_loader, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(val_loader)