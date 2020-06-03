"""Converters for pooling layers in Keras
"""
import onnx as O
import copy

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class MaxPooling2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    kernel_size = self.layer.pool_size
    strides = self.layer.strides
    # Construct padding array
    if self.node.extra_attr is None:
      if self.layer.padding == 'valid':
        padding_array = [0, 0, 0, 0]
      else:
        input_shape = self.input_shapes[0]
        padding_array = helper.getPadding(
          input_shape[2:4],
          kernel_size,
          self.layer.strides
        )
    else:
      if self.layer.padding == 'valid':
        padding_array = [self.node.extra_attr[0][0], self.node.extra_attr[1][0],
                         self.node.extra_attr[0][1], self.node.extra_attr[1][1]]
      else:
        input_shape = self.input_shapes[0]
        input_shape[2] += self.node.extra_attr[0][0] + self.node.extra_attr[0][1]
        input_shape[3] += self.node.extra_attr[1][0] + self.node.extra_attr[1][1]
        padding_array = helper.getPadding(
          input_shape[2:4],
          kernel_size,
          self.layer.strides
        )
        padding_array[0] += self.node.extra_attr[0][0]
        padding_array[1] += self.node.extra_attr[1][0]
        padding_array[2] += self.node.extra_attr[0][1]
        padding_array[3] += self.node.extra_attr[1][1]
    node = O.helper.make_node(
      op_type='MaxPool',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      kernel_shape=kernel_size,
      pads=padding_array,
      strides=strides
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
    actual_keras_shape = copy.copy(input_tree_tensor.keras_shape)
    if self.node.extra_attr is not None:
      actual_keras_shape[1] += self.node.extra_attr[0][0] + self.node.extra_attr[0][1]
      actual_keras_shape[2] += self.node.extra_attr[1][0] + self.node.extra_attr[1][1]
    output_keras_shape = self.layer.compute_output_shape(actual_keras_shape)
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    self.output_shape = output_tree_tensor.shape
    return output_value

class AveragePooling2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    kernel_size = self.layer.pool_size
    strides = self.layer.strides
    # Construct padding array
    if self.node.extra_attr is None:
      if self.layer.padding == 'valid':
        padding_array = [0, 0, 0, 0]
      else:
        input_shape = self.input_shapes[0]
        padding_array = helper.getPadding(
          input_shape[2:4],
          kernel_size,
          self.layer.strides
        )
    else:
      if self.layer.padding == 'valid':
        padding_array = [self.node.extra_attr[0][0], self.node.extra_attr[1][0],
                         self.node.extra_attr[0][1], self.node.extra_attr[1][1]]
      else:
        input_shape = self.input_shapes[0]
        input_shape[2] += self.node.extra_attr[0][0] + self.node.extra_attr[0][1]
        input_shape[3] += self.node.extra_attr[1][0] + self.node.extra_attr[1][1]
        padding_array = helper.getPadding(
          input_shape[2:4],
          kernel_size,
          self.layer.strides
        )
        padding_array[0] += self.node.extra_attr[0][0]
        padding_array[1] += self.node.extra_attr[1][0]
        padding_array[2] += self.node.extra_attr[0][1]
        padding_array[3] += self.node.extra_attr[1][1]
    node = O.helper.make_node(
      op_type='AveragePool',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      kernel_shape=kernel_size,
      pads=padding_array,
      strides=strides
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
    actual_keras_shape = copy.copy(input_tree_tensor.keras_shape)
    if self.node.extra_attr is not None:
      actual_keras_shape[1] += self.node.extra_attr[0][0] + self.node.extra_attr[0][1]
      actual_keras_shape[2] += self.node.extra_attr[1][0] + self.node.extra_attr[1][1]
    output_keras_shape = self.layer.compute_output_shape(actual_keras_shape)
    output_tree_tensor.set_shape(output_keras_shape)
    output_value = O.helper.make_tensor_value_info(
      output_tree_tensor.name,
      helper.dtype,
      output_tree_tensor.shape
    )
    self.output_shape = output_tree_tensor.shape
    return output_value

# GlobalAveragePooling2D layer
class GlobalAveragePooling2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Notice it should be channel first for input
    if hasattr(self.layer, "strides"):
      # This is acturally a averege pool layer.
      # And it follows a Flatten. No need to add ourself.
      node = O.helper.make_node(
        op_type='GlobalAveragePool',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
      )
      return [node], []
    # For other global average pool, add flatten.
    # The GlobalAvereagePool of Onnx output shape N C 1 1
    node = O.helper.make_node(
      op_type='GlobalAveragePool',
      inputs=self.inputs,
      outputs=[self.name + '_tmp'],
      name=self.name)
    node_list = [node]
    inner_shape = list(self.output_shape)
    for _ in range(2, len(self.layer.input_shape)):
      inner_shape.append(1)
    info = O.helper.make_tensor_value_info(
      self.name + '_tmp',
      helper.convertKerasType(helper.dtype),
      inner_shape
    )
    value_infos = [info]
    # Flatten the result to be the same shape of Keras output
    node = O.helper.make_node(
      op_type='Flatten',
      inputs=[self.name + '_tmp'],
      outputs=self.outputs,
      name=self.name + '_flatten',
      axis=1
      )
    node_list.append(node)
    return node_list, value_infos

# GlobalMaxPooling2D layer
class GlobalMaxPooling2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Notice it should be channel first for input
    # The GlobalAvereagePool of Onnx output shape N C 1 1
    node = O.helper.make_node(
      op_type='GlobalMaxPool',
      inputs=self.inputs,
      outputs=[self.name + '_tmp'],
      name=self.name)
    node_list = [node]
    inner_shape = list(self.output_shape)
    for _ in range(2, len(self.layer.input_shape)):
      inner_shape.append(1)
    info = O.helper.make_tensor_value_info(
      self.name + '_tmp',
      helper.convertKerasType(helper.dtype),
      inner_shape
    )
    value_infos = [info]
    # Flatten the result to be the same shape of Keras output
    node = O.helper.make_node(
      op_type='Flatten',
      inputs=[self.name + '_tmp'],
      outputs=self.outputs,
      name=self.name + '_flatten',
      axis=1
      )
    node_list.append(node)
    return node_list, value_infos
