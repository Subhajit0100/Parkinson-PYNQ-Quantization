import onnx
from onnx import version_converter

# Data paths
input_path = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8.onnx"
output_path = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_qat_int8_opset13.onnx"

print("Loading model...")
original_model = onnx.load(input_path)

# opset version
current_opset = original_model.opset_import[0].version
print(f"Current Opset Version: {current_opset}")

print("Converting version to Opset 13...")
try:
    #  To Opset 13
    converted_model = version_converter.convert_version(original_model, 13)


    onnx.save(converted_model, output_path)
    print(f"Successfully converted! Saved to: {output_path}")

except Exception as e:
    print(f"\nConversion failed due to: {e}")
