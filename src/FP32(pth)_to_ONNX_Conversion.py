import torch
import torch.nn as nn
from torchvision import models

# Force CPU mode for stable structural tracing
device = torch.device("cpu")
CHECKPOINT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_best.pth"
ONNX_OUTPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_fp32.onnx"

# ORIGINAL ARCHITECTURE
print("Rebuilding standard MobileNetV2 architecture...")
model_fp32 = models.mobilenet_v2(weights=None)

# trained classifier head
model_fp32.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model_fp32.last_channel, 2)
)

# FP32 WEIGHTS
print("Loading trained weights from checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model_fp32.load_state_dict(checkpoint["model_state_dict"], strict=True)

#evaluation 
model_fp32.eval()

# EXPORT FP32 ONNX
print(f"Exporting model to {ONNX_OUTPUT_PATH}...")

#  dummy input vector
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# FP32 graph
torch.onnx.export(
    model_fp32,                    
    dummy_input,                   
    ONNX_OUTPUT_PATH,               
    export_params=True,            
    opset_version=13,               
    do_constant_folding=True,       
    input_names=['input'],          
    output_names=['output'],        
    dynamic_axes={                  
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"\nSuccess! Pure FP32 ONNX model saved as '{ONNX_OUTPUT_PATH}'")
