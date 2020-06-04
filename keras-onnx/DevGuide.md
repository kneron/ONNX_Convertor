# Developer Guide

## Introduction

This guide will give your some idea before diving into the codes. The basic principle is that try to avoid assumptions. **DO NOT stage unnecessary generated files** in the git commits. Please read the notice carefully.

## NOTICE

1. Please **clear the outputs** before committing ipython notebooks.
2. **DO NOT** commit data large model files. If it is needed by certain test process, commit it using Git LFS.
3. Onnx CNTK backend does not support Reshape operation.
4. Onnx CNTK backend does not support BN with `epsilon != 1e-5`.
5. Use np.tolist() to convert numpy array to list instead of list().

## Assumptions

1. There is no loop inside the imported model.
2. Assume the whole model take only one `data_format`.
3. Assume the whole model take only one `data_type` for float.
4. Convert the model only for inference.
5. Assume shapes before and after Reshape layer have channels.
6. Only support RNN begin from Reshape or Input layers.
7. Only support RNN with only one start layer.

## Modifications

1. Duplicate shared layers.
2. Share weights for shared layers.
3. Extract the weight from the model into a node.
4. The pad layer only cares about the H*W part.
5. Weights have been made into 1D for tensors.
6. Treat None batch as 1. Treat None width and height as -1.
7. Replace Dense(FC) with Gemm.
8. GlobalAveragePooling is seperate into GlobalAveragePool and Flatten.

## General Structure

The files outside folder *onnx_keras* are simple wrappers and some small tools helping with generating code. Inside *onnx_keras*, The main control procedue is in *frontend.py*. Thus, if you want to start, please check this file to get a general idea first.

The general excuting sequence is descibed as following:

1. Load the Keras model using the Keras Python library.
2. Use the function defined in *preprocess.py* and the structures defined in *tree_structure.py* to construct a DAG to represent the model structure. Shared layers are duplicated during this step.
3. Travel through the DAG and convert each tree node in the DAG. The convertors for each type of layers are defined in *[name]_layer.py*, where name are replaced by the real type names.
4. Generate the value infos which hold the data between nodes.
5. Generate the onnx model using generated nodes and value infos.

## How to add a new layer converter

First of all, you need to know which file you will be editing. To add a new layer, we only need to edit one file and run a python script.

If the new converter's layer to add belongs to one of the existed basic types, then edit the related file. For example, if you want to add `Conv2D`, you should edit the `conv_layers.py`. If the related file of the layer type is not there, you can create the file yourself. For the new file, the file name should start with the type name and end with `_layers.py`. And you should as least import the following modules:

```python
import helper
from base_layer import Layer
from exceptions import FeatureNotImplemented, OnnxNotSupport
```

In the file, you should create a class for your new layer. Basically, the class for the layer should have the same name as in Keras and inherit the `Layer` class. And it shall as least have to function. It should be like:

```python
class Conv2D(Layer):
  def __init__(self, inputs, outname, layer, data_format='channels_last'):
    Layer.__init__(self, inputs, outname, layer, data_format)
  def generate(self):
    # Your code here to convert the Keras layer.
    return node_list, value_infos
```

The `node_list` is a list, which contains the generated nodes. The `value_infos` is a list which contains the value informations between nodes.

After your modification, you should run `refresh_layers.py` which generate a new `layer.py` to apply your changes.
