import os
import copy
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ==========================================================
# COLAB DRIVE MOUNT (Optional)
# ==========================================================
# If your dataset is stored on Google Drive, uncomment these two lines:
# from google.colab import drive
# drive.mount('/content/drive')

# ==========================================================
# FIXED RANDOM SEED
# ==========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 60)
print("Random Seed :", SEED)
print("=" * 60)

# ==========================================================
# DEVICE CONFIGURATION (Enforces High-Speed CUDA on Colab)
# ==========================================================
# Note: Make sure to set Colab runtime to GPU!
# (Runtime -> Change runtime type -> T4 GPU / L4 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device :", device)
if device.type == 'cuda':
    print("GPU Model    :", torch.cuda.get_device_name(0))
print("=" * 60)

# ==========================================================
# DATASET PATH (Update this to your Colab directory)
# ==========================================================
# Example if uploaded directly to Colab filesystem: '/content/Dataset_3'
# Example if using Google Drive: '/content/drive/MyDrive/PD_CNN/Dataset_3'
DATASET_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/Dataset_3"

# Dynamic Worker Allocation for Colab VM Environment
NUM_WORKERS = os.cpu_count()
print(f"Allocating Data Workers: {NUM_WORKERS}")

# ==========================================================
# TRAIN AUGMENTATION
# ==========================================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.08,0.08),
        scale=(0.90,1.10),
        shear=5
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.3
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.2,
        scale=(0.01,0.03),
        ratio=(0.5,2.0)
    ),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# VALIDATION
# ==========================================================
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# LOAD DATASET
# ==========================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset directory not found at {DATASET_PATH}. Please upload or verify your path.")

train_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=val_transform
)

print()
print("Classes :", train_dataset.classes)
print("Class Mapping :", train_dataset.class_to_idx)
print("Total Images :", len(train_dataset))
print()

# ==========================================================
# STRATIFIED 5 FOLD
# ==========================================================
targets = np.array(train_dataset.targets)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# ==========================================================
# BUILD MOBILENETV2
# ==========================================================
def build_model():
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    # Freeze all feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Fine tune only the last MobileNet blocks
    for param in model.features[14:].parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(
            model.last_channel,
            2
        )
    )

    return model.to(device)

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8

best_global_accuracy = 0.0
fold_results = []

# ==========================================================
# START 5-FOLD TRAINING
# ==========================================================
for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(targets)), targets), start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold} / 5")
    print("=" * 70)

    # ---------------------------------------
    # Compute class statistics
    # ---------------------------------------
    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets)
    print("Training Class Count :", class_counts)

    # ---------------------------------------
    # Weighted Random Sampler
    # ---------------------------------------
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = build_model()

    # --------------------------------------------
    # Loss Function
    # --------------------------------------------
    hc_count = class_counts[0]
    pd_count = class_counts[1]

    weights = torch.tensor([
        len(train_targets)/(2*hc_count),
        len(train_targets)/(2*pd_count)
    ], dtype=torch.float32).to(device)

    print("Loss Weights :", weights)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    # --------------------------------------------
    # Optimizer
    # --------------------------------------------
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------
    # Scheduler
    # --------------------------------------------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
        # verbose=False  <-- REMOVE OR DELETE THIS LINE COMPLETELY
    )

    best_fold_accuracy = 0.0
    best_fold_model = None
    early_stop_counter = 0

    # ======================================================
    # EPOCH LOOP
    # ======================================================
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # --------------------------------------------
        # TRAIN
        # --------------------------------------------
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train

        # ==================================================
        # VALIDATION
        # ==================================================
        model.eval()
        val_correct = 0
        val_total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total
        scheduler.step(val_accuracy)

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Loss {running_loss:.4f} | "
            f"Train {train_accuracy:.2f}% | "
            f"Val {val_accuracy:.2f}%"
        )

        # ==================================================
        # BEST MODEL OF THIS FOLD
        # ==================================================
        if val_accuracy > best_fold_accuracy:
            best_fold_accuracy = val_accuracy
            best_fold_model = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # ==================================================
        # EARLY STOPPING
        # ==================================================
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore best model of this fold
    model.load_state_dict(best_fold_model)

    # ==========================================================
    # FINAL VALIDATION USING BEST MODEL OF THIS FOLD
    # ==========================================================
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy()) # Added .cpu() fix for GPU execution
            y_pred.extend(preds.cpu().numpy())

    # ==========================================================
    # METRICS
    # ==========================================================
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\nConfusion Matrix")
    print(cm)
    print()
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"Balanced Accuracy  : {balanced_acc*100:.2f}%")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")

    fold_results.append(balanced_acc)

    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================
    if balanced_acc > best_global_accuracy:
        best_global_accuracy = balanced_acc
        torch.save({
            "fold": fold,
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx
        }, "mobilenetv2_best.pth")
        print("\nBest model saved.")

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": train_dataset.class_to_idx
}, "mobilenetv2_final.pth")

# ==========================================================
# FINAL RESULTS
# ==========================================================
print("\n" + "="*70)
print("Training Finished")
print("="*70)
print()

for i, score in enumerate(fold_results):
    print(f"Fold {i+1} Balanced Accuracy : {score*100:.2f}%")

print()
print(f"Average Balanced Accuracy : {np.mean(fold_results)*100:.2f}%")
print(f"Best Balanced Accuracy    : {best_global_accuracy*100:.2f}%")
print()
print("Saved Models:")
print("  mobilenetv2_best.pth")
print("  mobilenetv2_final.pth")
