import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# Use CUDA for fast QAT passes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
DATASET_PATH = r"/content/drive/MyDrive/Parkinsion_2d_data/Dataset_3"
CHECKPOINT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_best.pth"
BATCH_SIZE = 32
NUM_QAT_EPOCHS = 10

# ==========================================================
# DATA LOADING (Exact balanced original config)
# ==========================================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08,0.08), scale=(0.90,1.10), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transform)
targets = np.array(train_dataset.targets)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
best_fold = checkpoint["fold"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(targets)), targets), start=1):
    if fold == best_fold:
        target_train_idx = train_idx
        target_val_idx = val_idx
        break

train_targets = targets[target_train_idx]
class_counts = np.bincount(train_targets)
sample_weights = torch.DoubleTensor((1.0 / class_counts)[train_targets])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(Subset(train_dataset, target_train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=os.cpu_count(), pin_memory=True)
val_loader = DataLoader(Subset(val_dataset, target_val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

weights = torch.tensor([len(train_targets)/(2*class_counts[0]), len(train_targets)/(2*class_counts[1])], dtype=torch.float32).to(device)

# ==========================================================
# REBUILD FP32 MODEL & LOAD WEIGHTS
# ==========================================================
model_fp32 = models.mobilenet_v2(weights=None)
model_fp32.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model_fp32.last_channel, 2)
)
model_fp32.load_state_dict(checkpoint["model_state_dict"], strict=True)
model_fp32.eval()

# ==========================================================
# MODERN FX CONFIGURATION MAPPING (FIXES RUNTIMEERROR)
# ==========================================================
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping

print("\nConfiguring pure FX mapping engine...")

# Force uniform per_tensor schemes to prevent graph converter schema collisions
qat_qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')

# We build a strict mapping structure
qconfig_mapping = QConfigMapping().set_global(qat_qconfig)

# 1. Block early sensitive layers entirely from quantization processing
for i in range(14):
    qconfig_mapping.set_module_name(f"features.{i}", None)

example_inputs = (torch.randn(1, 3, 224, 224),)

# 2. Structure the active QAT graph instance cleanly
model_qat = quantize_fx.prepare_qat_fx(model_fp32, qconfig_mapping, example_inputs)

# 3. Explicitly manage gradients based on your layer targets
model_qat.train()
for name, param in model_qat.named_parameters():
    if any(f"features.{i}." in name for i in range(14)):
        param.requires_grad = False  # Freeze early features completely
    else:
        param.requires_grad = True   # Enable gradient adjustments on 14+ and head

model_qat.to(device)

# ==========================================================
# STRICT DUAL-ACCURACY QAT TRAINING LOOP
# ==========================================================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_qat.parameters()), lr=1e-6)

best_combined_score = 0.0
best_model_state = None

print(f"\nStarting QAT optimization...")
for epoch in range(NUM_QAT_EPOCHS):
    model_qat.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_qat(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = (correct / total) * 100

    # Validation Pass
    model_qat.eval()
    y_val_true, y_val_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model_qat(images.to(device))
            _, preds = torch.max(outputs, 1)
            y_val_true.extend(labels.numpy())
            y_val_pred.extend(preds.cpu().numpy())

    val_acc = accuracy_score(y_val_true, y_val_pred) * 100
    val_balanced_acc = balanced_accuracy_score(y_val_true, y_val_pred) * 100
    print(f"Epoch {epoch+1:02d} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Balanced Val Acc: {val_balanced_acc:.2f}%")

    # Balanced checkpointer capturing peak performance configurations
    combined_score = (val_acc + val_balanced_acc) / 2.0
    if combined_score > best_combined_score:
        best_combined_score = combined_score
        best_model_state = copy.deepcopy(model_qat.state_dict())

# Prepare for safe deployment compilation
model_qat.to("cpu")
model_qat.load_state_dict(best_model_state)
model_qat.eval()

# ==========================================================
# SAFE FX CONVERSION
# ==========================================================
print("\nConverting optimized QAT model into finalized INT8 representation...")
model_int8 = quantize_fx.convert_fx(model_qat)
print("Conversion Complete with no runtime errors!")

# Final evaluation
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model_int8(images.to("cpu"))
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print("\n" + "="*50)
print("             FINAL QUANTIZED METRICS")
print("="*50)
print(f"Final INT8 Accuracy          : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Final INT8 Balanced Accuracy : {balanced_accuracy_score(y_true, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# Save the production model file
torch.jit.save(torch.jit.script(model_int8), "mobilenetv2_qat_int8.pt")
print("\nEdge model saved as 'mobilenetv2_qat_int8.pt'")
