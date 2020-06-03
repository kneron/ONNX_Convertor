"""Converters for advanced activation layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .core_layers import Activation
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class LeakyReLU(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node = O.helper.make_node(
      'LeakyRelu',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      alpha=self.layer.alpha.tolist()
    )
    return [node], []

class ReLU(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    if self.layer.max_value is None and self.layer.threshold == 0:
      node = O.helper.make_node(
        op_type='Relu',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name
      )
    elif helper.compatibility:
      helper.logger.warning("Under compatibility mode. Generating Relu instead of Clip for layer %s.", self.name)
      node = O.helper.make_node(
        op_type='Relu',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        max=float(self.layer.max_value)
      )
    elif self.layer.max_value is None:
      node = O.helper.make_node(
        op_type='Clip',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        min=float(self.layer.threshold)
      )
    else:
      node = O.helper.make_node(
        op_type='Clip',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        min=float(self.layer.threshold),
        max=float(self.layer.max_value)
      )
    return [node], []

class PReLU(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Construct slope
    if helper.set_duplicate_weights:
      slope_name = self.name + "_scale"
    else:
      slope_name = self.layer.weights[0].name
    slope = self.layer.get_weights()[0]
    if slope_name not in helper.known_tensors:
      if self.layer.shared_axes is None:
        if helper.data_format == 'channels_last' and len(slope.shape) == 3:
          slope = np.transpose(slope, [2, 0, 1])
      else:
        if self.layer.shared_axes != [1, 2]:
          raise FeatureNotImplemented("PRelu shared axes " + str(self.layer.axes))
        elif len(slope.shape) > 1:
          slope = slope.reshape((slope.shape[-1], 1, 1))
    self.inputs.append(slope_name)
    node_list, values = helper.getConstantNodeByName(slope_name, slope)
    # Make the node
    node = O.helper.make_node(
      op_type='PRelu',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
      )
    node_list.append(node)
    return node_list, values

  # Softmax layer
class Softmax(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
    self.act_layer = Activation(node)
  def generate(self):
    return self.act_layer.softmax(self.layer.axis)

class Elu(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node = O.helper.make_node(
      'Elu',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      alpha=float(self.layer.alpha)
    )
    return [node], []
