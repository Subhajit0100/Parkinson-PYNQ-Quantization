import torch
import torch.nn as nn
from torchvision import models

# Force CPU mode for stable structural tracing
device = torch.device("cpu")
CHECKPOINT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_best.pth"
ONNX_OUTPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_fp32.onnx"

# ==========================================================
# 1. REBUILD ORIGINAL ARCHITECTURE
# ==========================================================
print("Rebuilding standard MobileNetV2 architecture...")
model_fp32 = models.mobilenet_v2(weights=None)

# Replicate your exact trained classifier head
model_fp32.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model_fp32.last_channel, 2)
)

# ==========================================================
# 2. LOAD YOUR HIGH-ACCURACY FP32 WEIGHTS
# ==========================================================
print("Loading trained weights from checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model_fp32.load_state_dict(checkpoint["model_state_dict"], strict=True)

# Set model to evaluation mode (deactivates dropout layers)
model_fp32.eval()

# ==========================================================
# 3. EXPORT CLEANLY TO FP32 ONNX
# ==========================================================
print(f"Exporting model to {ONNX_OUTPUT_PATH}...")

# Create a dummy input vector matching your exact input shape (Batch, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# Export standard FP32 graph
torch.onnx.export(
    model_fp32,                     # Your standard high-accuracy PyTorch model
    dummy_input,                    # Dummy input matrix shape
    ONNX_OUTPUT_PATH,               # Target file output path
    export_params=True,             # Embed the floating-point weights inside the file
    opset_version=13,               # High compatibility version for ONNX runtimes
    do_constant_folding=True,       # Pre-computes structural constants for optimization
    input_names=['input'],          # Name of the input tensor
    output_names=['output'],        # Name of the output tensor
    dynamic_axes={                  # Allows your code to accept variable batch sizes later
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"\nSuccess! Pure FP32 ONNX model saved as '{ONNX_OUTPUT_PATH}'")
