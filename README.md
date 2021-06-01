# ONNX Convertors

## Introduction

This project include converters and optimizers in Python3 that are used to generate and optimize ONNX models for Kneron Toolchain.

The current using onnx version is 1.6.0 which is under operator set version 11.

Note that the generated onnx is **especially optimized for Kneron Toolchain**, which may not be the best solution for general purpose ONNX usage, though in most cases, math optimal solutions are good for the toolchain. For example, the ONNX to ONNX optimizer has an option on add a do-nothing BatchNormalization node on the skip branch. This seems not optimal from the math perspective. However, for the Kneron Toolchain, adding such a layer can improve its quantization process.

## Keras to ONNX

The Keras to ONNX converter support convert Keras （version 2.2.4) model to onnx. For detail, please check folder `keras-onnx`.

Note that what we support here is Keras, not `tf.Keras`. `tf.Keras` is actually a part of Tensorflow.

## Caffe to ONNX

The Caffe to ONNX converter support convert Caffe to ONNX. Since Caffe is an old platform which is using C++, our support for the Caffe is based on [`intel-caffe`](https://github.com/intel/caffe). We are currently using version 1.1.3. You may need to test other versions yourself.  Customized operators may not be supported. Please check folder` caffe-onnx`.

## Pytorch to ONNX

Pytorch to onnx is achieved through the combination of the [`torch.onnx`](https://pytorch.org/docs/stable/onnx.html) and our optimizer under `optimizer_scripts` folder. Note that you may need to specify opset to 11 while you are exporting the model using `torch.onnx`. Please check `optimizer_scripts` folder. We are currently using version 1.7.1. Other version's which has `torch.onnx` may also works. But versions earlier than 1.1.0 are not recommended.

## Tensorflow to ONNX

Tensorflow to onnx is based on the open source project [`tensorflow-onnx`](https://github.com/onnx/tensorflow-onnx). Our `tensorflow2onnx.py` use the open source project to generate the ONNX file and optimize the generated ONNX. Currently, the Tensorflow model support is pretty limited. It is still under development. Please check `optimizer_scripts` folder. We are currently using version 1.6.0. You can use the latest version of `tensorflow-onnx`. Just remember to set the onnx opset version to 11.

## Optimizers

General optimizer optimize ONNX model for the Kneron Toolchain use. Please check `optimizer_scripts` folder.

## Development Guide.

Please utilize the Github merge request system.

If you have any question about the project, please contact <jiyuan@kneron.us>.
