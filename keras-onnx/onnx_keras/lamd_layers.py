""" This layers is for debug purpose only
"""

import sys
import onnx as O
from .base_layer import Layer
from . import helper

class Lambda(Layer):
  def __init__(self, node):
    if node.name in helper.custom_name2type:
      helper.logger.debug("Found type classification for lambda layer " + node.name)
      self.type = helper.custom_name2type[node.name]
      self.opid = helper.custom_type2opid[self.type]
    else:
      helper.logger.warning("Lambda layer " + node.name + " is not properly defined in custom.json. Converting to Mystery layer.")
      self.opid = helper.opid_counter
      helper.opid_counter += 1
      self.type = "CustomOP" + str(self.opid)
    Layer.__init__(self, node)
  def generate(self):
    node = O.helper.make_node(
      op_type='Mystery',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      type=self.type,
      opid=self.opid
    )
    return [node], []
  def setOutputValue(self):
    """Set up the current layer output tensor.

    # Return value
    The corresponding value_info
    """
    # Construct output value info
    input_tree_tensor = self.node.inputs[0]
    output_tree_tensor = self.node.outputs[0]
    # For lambda layer, we check if they have a valid input/output size.
    # If so, we will use it directly.
    # Otherwise, we assume the output size is the same as the input size.
    input_keras_shape = input_tree_tensor.keras_shape
    model_keras_shape = list(self.layer.input_shape)
    if model_keras_shape[0] is None:
      model_keras_shape[0] = 1
    if model_keras_shape == input_keras_shape:
      helper.logger.debug("Using the output shape in lambda layer")
      output_keras_shape = list(self.layer.output_shape)
      if output_keras_shape[0] is None:
        output_keras_shape[0] = 1
    else:
      helper.logger.debug("Set output size the same as input size in lambda layer")
      output_keras_shape = input_keras_shape
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    self.output_shape = output_tree_tensor.shape
    return output_value
