

import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

TRAIN_TXT = "dataset/train.txt"
VAL_TXT = "dataset/val.txt"
TEST_TXT = "dataset/test.txt"
CLASSES_TXT = "dataset/classes.txt"


BATCH_SIZE = 16
IMG_SIZE = 224
LR = 2e-4
EPOCHS = 150
NUM_WORKERS = 4  # adjust according to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
classes = pd.read_csv(CLASSES_TXT, sep=" ", header=None, names=["label", "name"])
NUM_CLASSES = len(classes)

# ----------------------------
# READ TXT FILES
# ----------------------------
train_df = pd.read_csv(TRAIN_TXT, sep=" ", header=None, names=["image_path", "label"])
val_df = pd.read_csv(VAL_TXT, sep=" ", header=None, names=["image_path", "label"])
test_df = pd.read_csv(TEST_TXT, sep=" ", header=None, names=["image_path", "label"])

# ----------------------------
# DATASET
# ----------------------------
class InsectDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row.label), row.image_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.transforms:
            image = self.transforms(image=image)["image"]
        
        label = torch.tensor(row.label, dtype=torch.long)
        return image, label

# ----------------------------
# AUGMENTATIONS
# ----------------------------
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomRotate90(p=0.3),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# ----------------------------
# DATA LOADERS
# ----------------------------
train_dataset = InsectDataset(train_df, TRAIN_DIR, transforms=train_transform)
val_dataset = InsectDataset(val_df, VAL_DIR, transforms=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ----------------------------
# MODEL
# ----------------------------
model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ----------------------------
# LOSS & OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ----------------------------
# TRAIN & VALIDATION FUNCTIONS
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader)
    for images, labels in loop:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_postfix(loss=running_loss/total, acc=correct/total)
    
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss/total, correct/total

# ----------------------------
# TRAIN LOOP
# ----------------------------
if __name__ == "__main__":
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "efficientnetv2s_best.pth")
            print("Saved best model!")
    print("Training complete.")
