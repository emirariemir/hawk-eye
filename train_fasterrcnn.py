import os
import torch
import torchvision
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---- Load class names from data.yaml ----
def load_yaml_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# ---- Custom Dataset class for YOLO-format ----
class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_files = sorted(os.listdir(images_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load and normalize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img_tensor = F.to_tensor(img)

        # Parse YOLO label
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = (x - bw / 2) * w
                y1 = (y - bh / 2) * h
                x2 = (x + bw / 2) * w
                y2 = (y + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return img_tensor, target

# ---- Load model ----
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---- Training loop ----
def train(model, dataloader, device, num_epochs=10, lr=0.005):
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ---- Main entry ----
if __name__ == "__main__":
    # Edit these paths
    data_yaml_path = "vehicle_archive/data.yaml"
    train_img_dir = "vehicle_archive/train/images"
    train_lbl_dir = "vehicle_archive/train/labels"

    class_names = load_yaml_classes(data_yaml_path)
    num_classes = len(class_names) + 1  # +1 for background class

    dataset = YoloDataset(train_img_dir, train_lbl_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, dataloader, device, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), "fasterrcnn_yolo_format.pth")
