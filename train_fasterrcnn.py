import os
import cv2
from evaluate_rcnn import compute_validation_loss
import torch
import torchvision
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---- COCO-style Dataset Class ----
class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img)

        # Load boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        return img_tensor, target

# ---- Load Faster R-CNN model ----
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---- Training Loop ----
def train(model, train_loader, val_loader, device, num_epochs=10, lr=0.005):
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        model.eval()
        with torch.no_grad():
            val_loss = compute_validation_loss(model, val_loader, device)
            print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# ---- Main Entry ----
if __name__ == "__main__":
    train_img_dir = "dataset/train_reduced/images"
    train_ann_file = "dataset/train_reduced/annotations.json"

    val_img_dir = "dataset/train_reduced/images"
    val_ann_file = "dataset/train_reduced/annotations.json"

    # Load COCO to count categories
    coco = COCO(train_ann_file)
    num_classes = len(coco.getCatIds()) + 1  # +1 for background

    dataset = CocoDetectionDataset(train_img_dir, train_ann_file)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    val_dataset = CocoDetectionDataset(val_img_dir, val_ann_file)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)

    train(model, train_loader, val_loader, device, num_epochs=50)

    # Save the model
    torch.save(model.state_dict(), "fasterrcnn_coco_trained.pth")

