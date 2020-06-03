# Developer Guide

## Introduction

This guide will give your some idea before diving into the codes. The basic principle is that try to avoid assumptions. **DO NOT stage unnecessary generated files** in the git commits. Please read the notice carefully.

## NOTICE

1. Please **clear the outputs** before committing ipython notebooks.
2. **DO NOT** commit data large model files.
3. Onnx CNTK backend does not support Reshape operation.
4. Onnx CNTK backend does not support BN with `epsilon != 1e-5`.
5. Use np.tolist() to convert numpy array to list instead of list().

## Assumptions

1. All the Keras layer names are unique.
2. There is no loop inside the imported model.
3. Assume the whole model take only one `data_type` for float.
4. Convert the model only for inference.
5. No activation inside layers.
6. BatchNormalizaion layer can only deal with none or 1 batch size.

## Modifications

1. Duplicate shared layers to meet the SSA requirement
2. Seperate Keras output node from different layers.
3. Extract the weight from the model into a node.
4. The pad layer only cares about the H*W part.
5. Weights have been made into 1D for tensors
6. Treat None batch as 1. Treat None width and height as 32.
7. Replace Dense(FC) with Gemm.
8. GlobalAveragePooling is seperate into GlobalAveragePool and Flatten.
9. BatchNormalization with 2 input dimensions will be expanded into 4D.

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
  def __init__(self, inputs, outname, layer):
    Layer.__init__(self, inputs, outname, layer)
  def generate(self):
    # Your code here to convert the Keras layer.
    return node_list, value_infos
```

The `node_list` is a list, which contains the generated nodes. The `value_infos` is a list which contains the value informations between nodes.

After your modification, you should run `generate_layers.py` which generate a new `layer.py` to apply your changes.
