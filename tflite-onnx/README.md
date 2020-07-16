# Converter for TFlite to Onnx

## Description

Convert an TFlite model to Onnx.
We parse TFlite model use:
1. flatc(https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html) 
2. tensorflow lite schema file(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs)

### Prerequisites

We test these script using:
* **Python 3**:    Python==3.7.7
* **ONNX**:    onnx==1.4.1
* **Tensorflow**:    tensorflow==1.15.0

## Basic Usage

1. run `python path/to/generate_onnx.py -h` to check the parameter.
* 1.1. -tflite: the path to the tflite file
* 1.2. -save_path: final onnx file path to save
2. run `python generated_onnx.py -tflite YOUR_TFLITE_PATH -save_path ONNX_SAVE_PATH` to convert the model,

## Tested Model Table

### tensorflow model zoo:
#### https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
* Mobile_models/ssd_mobiledet_cpu_coco
* Mobile_models/ssd_mobilenet_v3_large_coco
* Mobile_models/ssd_mobilenet_v3_small_coco

### tflite model hub:
#### https://www.tensorflow.org/lite/guide/hosted_models
* Mobilenet_V1 series
* Mobilenet_V2 series
* Inception_V3
* SqueezeNet
* DenseNet


## Now Supported Operators List
* ADD
* MUL
* PAD
* RESHAPE
* FULLY_CONNECTED
* CONV_2D
* DEPTHWISE_CONV_2D
* SOFTMAX
* RELU
* RELU6
* LOGISTIC
* CONCATENATION
* MEAN
* MAX_POOL_2D
* AVERAGE_POOL_2D
* SQUEEZE

## Example
1. Convert tflite to onnx 
* python generate_onnx.py -tflite ./example/example.tflite -save_path ./example -release_mode True
2. Convert onnx to the onnx which is optimized for Kneron Toolchain
* python ../optimizer_scripts/onnx2onnx.py ./example/example.onnx