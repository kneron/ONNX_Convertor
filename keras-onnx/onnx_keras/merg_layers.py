"""Converters for pooling layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class Add(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    if len(self.inputs) == 2:
      node = O.helper.make_node(
        op_type='Add',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name)
    else:
      node = O.helper.make_node(
        op_type='Sum',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name)
    return [node], []
  def setOutputValue(self):
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    output_keras_shape = input_tree_tensor.keras_shape
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    return output_value

class Subtract(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    if len(self.inputs) == 2:
      node = O.helper.make_node(
        op_type='Sub',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name)
    else:
      node = O.helper.make_node(
        op_type='Sub',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name)
    return [node], []
  def setOutputValue(self):
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    output_keras_shape = input_tree_tensor.keras_shape
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    return output_value

class Multiply(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Here we use BN to support mul with constant.
    input_a = self.node.inputs[0].input
    input_b = self.node.inputs[1].input
    if input_a.type != 'Constant' and input_b.type == 'Constant':
      flow_input = input_a
      data_input = input_b
    elif input_a.type == 'Constant' and input_b.type != 'Constant':
      flow_input = input_b
      data_input = input_a
    else:
      # For other elementwise multiply, just put mul here.
      #TODO: Add optimization for multiply with two constants
      node = O.helper.make_node(
        op_type='Mul',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name)
      return [node], []
    #NOTE This function has not been tested yet
    # Construct a bn
    inputs = [flow_input.outputs[0].name, data_input.outputs[0].name]
    shape = data_input.outputs[0].keras_shape[0]
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    ones_name = self.name + "_ones"
    zeros_name = self.name + "_zeros"
    ones_nodes, ones_info = helper.constructConstantNode(ones_name, ones)
    zeros_nodes, zeros_info = helper.constructConstantNode(zeros_name, zeros)
    inputs.append(zeros_name)
    inputs.append(zeros_name)
    inputs.append(ones_name)
    node = O.helper.make_node(
      op_type='BatchNormalization',
      inputs=inputs,
      outputs=self.outputs,
      name=self.name
    )
    nodes = [node]
    nodes += ones_nodes
    nodes += zeros_nodes
    infos = []
    infos += ones_info
    infos += zeros_info
    return nodes, infos
  def setOutputValue(self):
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    output_keras_shape = input_tree_tensor.keras_shape
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    return output_value

class Concatenate(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    if helper.data_format == 'channels_last' and len(list(self.layer.input_shape)):
      if self.layer.axis == -1 or self.layer.axis == len(list(self.layer.input_shape[0])) - 1:
        self.axis = 1
      elif self.layer.axis == 0:
        self.axis = 0
      else:
        self.axis = self.layer.axis + 1
    else:
      self.axis = self.layer.axis
    node = O.helper.make_node(
      op_type='Concat',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      axis=self.axis
    )
    return [node], []
  def setOutputValue(self):
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    output_keras_shape = list(input_tree_tensor.keras_shape)
    output_keras_shape[self.layer.axis] = 0
    for input_tree_tensor in self.node.inputs:
      output_keras_shape[self.layer.axis] += input_tree_tensor.keras_shape[self.layer.axis]
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    return output_value