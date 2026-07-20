import torch
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cpu")

PT_INPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8.pt"
ONNX_OUTPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8.onnx"

# 1. Load the TorchScript JIT graph
print(f"Loading TorchScript model from: {PT_INPUT_PATH}")
model_jit = torch.jit.load(PT_INPUT_PATH, map_location=device)
model_jit.eval()

# 2. Create input tensor blueprint
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

print(f"Forcing Legacy ONNX Serialization pathway to: {ONNX_OUTPUT_PATH}...")
try:
   
    torch.onnx.utils.export(
        model_jit,                  # TorchScript module
        (dummy_input,),             # Input wrapped as a tuple
        ONNX_OUTPUT_PATH,           # Output path
        export_params=True,         # Retain constant weights
        opset_version=13,           # Highly stable version for legacy tracing
        do_constant_folding=True,   # Optimize static constants
        input_names=['input'],
        output_names=['output']
    )
    print(f" Success! Your optimized INT8 ONNX model is successfully saved at:\n{ONNX_OUTPUT_PATH}")

except Exception as e:
    print(f"\nExport failed. Details: {e}")
