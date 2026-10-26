# Parkinson's Disease Detection from Hand-Drawn Images

A research prototype for classifying hand-drawn spiral and meander images using MobileNetV2 transfer learning. The repository contains scripts for model training, quantization-aware training (QAT), and PyTorch-to-ONNX conversion.

## Overview

The workflow trains a two-class image classifier, prepares an INT8 model for edge deployment, and exports FP32 and INT8 models to ONNX.

The accompanying paper describes deployment on a PYNQ-Z2 board. It reports a camera-based pipeline with image preprocessing and ONNX Runtime inference. The scripts shown in this repository cover model training and conversion; a live camera capture and inference application is not included in the listed source files.

## Repository contents

| File | Purpose |
|---|---|
| `src/MobileNetV2_Training.py` | Trains and validates MobileNetV2 using five-fold stratified cross-validation. Saves best and final checkpoints. |
| `src/Quantization_best_model.py` | Loads the best checkpoint, performs QAT, converts the model to INT8, and saves a TorchScript model. |
| `src/FP32(pth)_to_ONNX_Conversion.py` | Exports the trained FP32 checkpoint to ONNX. |
| `src/INT8(pth)_to_ONNX_conversion.py` | Exports the quantized TorchScript model to ONNX. |
| `src/FP32(ONNX)_to_PNYQ_FP32(ONNX)_converter.py` | Converts the FP32 ONNX model to opset 13. |
| `src/INT8(ONNX)_to_PYNQ_INT8(ONNX)_conversion.py` | Converts the INT8 ONNX model to opset 13. |

## Dataset

The paper evaluates the approach on HandPD, a dataset of spiral and meander drawings from healthy controls and people with Parkinson's disease.
Arrange your dataset in class subfolders, for example:

    Dataset_3/
    ├── Healthy/
    │   ├── image_001.png
    │   └── ...
    └── Parkinson/
        ├── image_001.png
        └── ...

Set `DATASET_PATH` in `MobileNetV2_Training.py` to the dataset directory. The script converts images to grayscale, resizes them to 224 × 224 pixels, replicates grayscale across three channels, and applies training-time augmentation.

## Requirements

The scripts use:

- Python
- PyTorch and torchvision
- NumPy
- scikit-learn
- ONNX

Install a PyTorch and torchvision build compatible with your environment, such as the Google Colab runtime you plan to use. The repository does not include a pinned dependency file.

## Training and export

Before running the scripts, update the dataset, checkpoint, and output paths near the top of each file. The defaults are absolute Google Colab paths.

The training script saves `mobilenetv2_best.pth` and `mobilenetv2_final.pth` in its current working directory. The QAT and conversion scripts expect some inputs under `/content/drive/MyDrive/Parkinsion_2d_data/`. Either copy the generated files to those locations or update the corresponding path constants.

Run the workflow in this order:

    python src/MobileNetV2_Training.py
    python src/Quantization_best_model.py
    python "src/FP32(pth)_to_ONNX_Conversion.py"
    python "src/INT8(pth)_to_ONNX_conversion.py"
    python "src/FP32(ONNX)_to_PNYQ_FP32(ONNX)_converter.py"
    python "src/INT8(ONNX)_to_PYNQ_INT8(ONNX)_conversion.py"

The optional Google Drive mount snippet in `MobileNetV2_Training.py` is commented out. In Colab, enable and configure it if your dataset or checkpoints are stored on Drive.

## Deployment described in the paper

The paper describes a PYNQ-Z2 setup using a webcam, image preprocessing, region-of-interest cropping, and ONNX Runtime with the CPU execution provider. The reported preprocessing includes grayscale conversion, bilateral filtering, CLAHE contrast enhancement, thresholding, ROI cropping, and resizing before inference.

The repository scripts prepare model files for this workflow. The board-side capture, preprocessing, and live inference application would need to be supplied separately.

## Reported results

The accompanying paper reports:

- On its HandPD evaluation, 95.95% accuracy for FP32 and 92.57% for QAT INT8.
- On a separate 76-image clinical collection evaluated on PYNQ-Z2, 96.05% accuracy for FP32 and 94.74% for INT8.
- Model size reduction from 9.15 MB to 2.44 MB for the quantized model.
- PYNQ-Z2 execution latency of approximately 1,367 ms for FP32 and 1,254 ms for INT8.

These are results reported in the paper; they may not reproduce without the same data splits, software versions, preprocessing, and hardware setup.

## Intended use

This repository is for research and prototyping. It is not a medical device and should not be used as a substitute for clinical assessment or diagnosis.
