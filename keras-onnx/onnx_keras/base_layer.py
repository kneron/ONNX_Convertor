"""Base layer class
"""
import logging
import onnx as O
from . import helper
from .exceptions import FeatureNotImplemented, OnnxNotSupport

class Layer:
  def __init__(self, node):
    self.node = node
    self.layer = node.klayer
    self.name = node.name
    self.inputs = self.getInputs()
    self.outputs = self.getOutputs()
    self.logger = logging.getLogger("onnx-keras")
    self.logger.setLevel(logging.DEBUG)
    self.output_shape = None
    self.input_shapes = []
    self.input_keras_shapes = []
    for input_tensor in self.node.inputs:
      self.input_shapes.append(input_tensor.shape)
      self.input_keras_shapes.append(input_tensor.keras_shape)

  def getInputs(self):
    """Generate a list of the input tensor names
    """
    input_list = []
    for in_tensor in self.node.inputs:
      input_list.append(in_tensor.name)
    return input_list

  def getOutputs(self):
    """Generate a list of the output tensor names
    """
    output_list = []
    for out_tensor in self.node.outputs:
      output_list.append(out_tensor.name)
    if len(output_list) > 1:
      raise OnnxNotSupport("More than 1 output in " + self.name)
    return output_list

  def generate(self):
    """Generate the nodes according to the original layer

    # Return value
    node_list   : a list of nodes generated
    value_infos : value_infos between nodes
    """
    return [], []

  def setOutputValue(self):
    """Set up the current layer output tensor.

    # Return value
    The corresponding value_info
    """
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    output_keras_shape = self.layer.compute_output_shape(tuple(input_tree_tensor.keras_shape))
    if self.node == helper.RNN_start_node:
      output_tree_tensor.set_shape(output_keras_shape, output_keras_shape)
    else:
      output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    self.output_shape = output_tree_tensor.shape
    return output_value
