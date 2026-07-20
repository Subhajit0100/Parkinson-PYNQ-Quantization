import onnx
from onnx import version_converter

# Define paths
input_path = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_fp32.onnx"
output_path = "/content/drive/MyDrive/Parkinsion_2d_data/mobilenetv2_fp32_opset13.onnx"

print("Loading model...")
original_model = onnx.load(input_path)

# Verify the current opset version
current_opset = original_model.opset_import[0].version
print(f"Current Opset Version: {current_opset}")

print("Converting version to Opset 13...")
try:
    # Downgrade the model to Opset 13
    converted_model = version_converter.convert_version(original_model, 13)

    # Save the new version
    onnx.save(converted_model, output_path)
    print(f"Successfully converted! Saved to: {output_path}")

except Exception as e:
    print(f"\nConversion failed due to: {e}")
