# Converter for TFlite to Onnx

## Description

Convert an TFlite model to Onnx.
We parse TFlite model use:
1. flatc(https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html) 
2. tensorflow lite schema file(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs)

### Prerequisites

We test our script using:
* **Python 3**:    Python==3.7.7
* **ONNX**:    onnx==1.6
* **Tensorflow**:    tensorflow==1.15.0
* **FlatBuffer**:    flatbuffers==1.12
* **Python-igraph**:    python-igraph==0.8.3

## Basic Usage

1. run `python path/to/tflite2onnx.py -h` to check the parameter.
* 1.1. -tflite: the path to the tflite file
* 1.2. -save_path: final onnx file path to save
2. run `python tflite2onnx.py -tflite YOUR_TFLITE_PATH -save_path ONNX_SAVE_PATH` to convert the model,

## Tested Model Table

### tensorflow object detection model zoo:
#### https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
#### ( use export_tflite_ssd_graph.py with --add_postprocessing_op=false )
* Mobile_models/ssd_mobiledet_cpu_coco
* Mobile_models/ssd_mobilenet_v3_large_coco
* Mobile_models/ssd_mobilenet_v3_small_coco
* Mobile_models/ssd_mobilenet_v2_mnasfpn_coco

### tflite model hub (Floating point models):
#### https://www.tensorflow.org/lite/guide/hosted_models
* Mobilenet_V1 series
* Mobilenet_V2 series
* Inception_V3
* Inception_V4
* Inception_ResNet_V2
* SqueezeNet
* DenseNet
* ResNet_V2_101
* EfficientNet-Lite


## Now Supported Operators List
* ADD
* AVERAGE_POOL_2D
* CONCATENATION
* CONV_2D
* DEPTH_TO_SPACE
* DEPTHWISE_CONV_2D
* ELU
* FULLY_CONNECTED
* L2_NORMALIZATION
* LEAKY_RELU
* LOGISTIC
* MUL
* MEAN
* MAXIMUM
* MAX_POOL_2D
* PAD
* PRELU
* RELU
* RELU6
* RESHAPE
* RESIZE_BILINEAR
* RESIZE_NEAREST_NEIGHBOR
* SOFTMAX
* SPACE_TO_DEPTH
* SQUEEZE
* TRANSPOSE_CONV


## Example 1: Convert model to the onnx optimized for Kneron Toolchain
#### Step0 (Optional). Download example tflite model
###### (Our provided example model is LFS-tracked file, sometimes 'git lfs pull' is needed)
    git lfs pull
#### Step1. Convert tflite to onnx  
###### (Originally, we add transpose nodes for the channel order difference between onnx and tflite. Use '-release_mode True' could ignore those transpose nodes)
    python ./onnx_tflite/tflite2onnx.py -tflite ./example/example.tflite -save_path ./example/example.onnx -release_mode True
#### Step2. Convert onnx to the onnx which is optimized for Kneron Toolchain
    python ../optimizer_scripts/onnx2onnx.py ./example/example.onnx


## Example 2: Convert part of tflite model to onnx
    python ./onnx_tflite/tflite2onnx.py -tflite ./example/example.tflite -save_path ./example/example.onnx -release_mode True -bottom_nodes sequential_1/model1/block_14_add/add


## *Note 1:
* "./example/example.tflite" and "./onnx_tflite/flatc/flatc" are large file and not necessary. So we let them being tracked with GIT-LFS. You can use 'git lfs pull' to download them.

## *Note 2 (convert tensorflow to tflite):
* Here is the official guide to convert tensorflow model and keras model to tflite:
    https://www.tensorflow.org/lite/guide/get_started#tensorflow_lite_converter

    Note that current tflite2onnx only support "float32" tflite,and current official tflite not support all tensorflow op  
    ( ref: https://www.tensorflow.org/lite/guide/get_started#ops_compatibility )  
    you might need to cut some unsupported nodes. 

