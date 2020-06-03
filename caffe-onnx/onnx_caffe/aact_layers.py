"""Converters for advanced activation layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class ReLU(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node = []
    if self.proto.relu_param.negative_slope == 0:
      node = O.helper.make_node(
        op_type='Relu',
        inputs = self.inputs,
        outputs = self.outputs,
        name = self.name
      )
    else:
      node = O.helper.make_node(
        op_type='LeakyRelu',
        inputs = self.inputs,
        outputs = self.outputs,
        name = self.name,
        alpha = self.proto.relu_param.negative_slope
      )
    return [node], []

class PReLU(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    value_infos = []
    w = self.layer.blobs[0].data
    wnode_name = self.name + '_weight'
    tn, ti = helper.constructConstantNode(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    node = O.helper.make_node(
      op_type='PRelu',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name
    )
    node_list.append(node)
    return node_list, value_infos

class Softmax(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node = O.helper.make_node(
      op_type = 'Softmax',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name,
      axis = 1
    )
    return [node], []

class Sigmoid(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node = O.helper.make_node(
      op_type = 'Sigmoid',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name
    )
    return [node], []