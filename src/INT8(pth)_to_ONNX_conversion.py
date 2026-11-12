import torch
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cpu")

PT_INPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8.pt"
ONNX_OUTPUT_PATH = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8.onnx"

print(f"Loading TorchScript model from: {PT_INPUT_PATH}")
model_jit = torch.jit.load(PT_INPUT_PATH, map_location=device)
model_jit.eval()


dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

print(f"Forcing Legacy ONNX Serialization pathway to: {ONNX_OUTPUT_PATH}...")
try:
   
    torch.onnx.utils.export(
        model_jit,                  
        (dummy_input,),             
        ONNX_OUTPUT_PATH,           
        export_params=True,         
        opset_version=13,           
        do_constant_folding=True,  
        input_names=['input'],
        output_names=['output']
    )
    print(f" Success! Your optimized INT8 ONNX model is successfully saved at:\n{ONNX_OUTPUT_PATH}")

except Exception as e:
    print(f"\nExport failed. Details: {e}")
