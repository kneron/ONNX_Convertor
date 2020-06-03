"""Converters for recurrent layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .core_layers import Activation
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

activation_mapping = {
  'elu': "Elu",
  'softplus': 'Softplus',
  'softsign': 'Softsign',
  'relu': 'Relu',
  'tanh': 'Tanh',
  'sigmoid': 'Sigmoid',
  'hard_sigmoid': 'HardSigmoid',
  'linear': None
}

class LSTM(Layer):
  """LSTM layer converter. This LSTM is not bidirectional.
  A LSTM layer will be converted into four layers:
  Transpose => LSTM => Transpose => Reshape
  """
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Convert LSTM
    node_list = []
    value_list = []
    # Keras activation layer
    keras_activation = self.layer.cell.activation.__name__
    if keras_activation not in activation_mapping:
      raise OnnxNotSupport("Activation in LSTM: " + keras_activation)
    keras_activation = activation_mapping[keras_activation]
    keras_recurrent_activation = self.layer.cell.recurrent_activation.__name__
    if keras_recurrent_activation not in activation_mapping:
      raise OnnxNotSupport("Activation in LSTM: " + keras_recurrent_activation)
    keras_recurrent_activation = activation_mapping[keras_recurrent_activation]
    activations = [keras_recurrent_activation, keras_activation, keras_activation]
    # Give activation alpha and beta
    activation_alpha = []
    activation_beta = []
    for activation in activations:
      if activation == 'Elu':
        activation_alpha.append(1.0)
        activation_beta.append(0.0)
      elif activation == 'HardSigmoid':
        activation_alpha.append(0.2)
        activation_beta.append(0.5)
      else:
        activation_alpha.append(0.0)
        activation_beta.append(0.0)
    # Direction
    if self.layer.go_backwards:
      direction = 'reverse'
    else:
      direction = 'forward'
    # Construct First Transpose
    input_list = []
    dims = [1, 0, 2]
    preprocess_name = self.name + "_preprocess"
    preprocess_node = O.helper.make_node(
      'Transpose',
      inputs=self.inputs,
      outputs=[preprocess_name],
      name=preprocess_name,
      perm=dims
    )
    preprocess_size = [self.node.inputs[0].keras_shape[1],
        self.node.inputs[0].keras_shape[0], self.node.inputs[0].keras_shape[2]]
    preprocess_info = O.helper.make_tensor_value_info(
        preprocess_name,
        helper.dtype,
        preprocess_size
    )
    node_list.append(preprocess_node)
    value_list.append(preprocess_info)
    input_list.append(preprocess_name)
    # Construct Weights
    w = self.layer.cell.get_weights()[0]
    w = np.transpose(w, [1, 0])
    w = np.expand_dims(w, 0)
    if helper.duplicate_weights:
      w_name = self.name + "_weight"
    else:
      w_name = self.layer.weights[0].name
    tn, ti = helper.getConstantNodeByName(w_name, w)
    node_list += tn
    value_list += ti
    input_list.append(w_name)
    # Construct recurrent weight
    rw = self.layer.cell.get_weights()[1]
    rw = np.transpose(rw, [1, 0])
    rw = np.expand_dims(rw, 0)
    if helper.duplicate_weights:
      rw_name = self.name + "_recurrent_weight"
    else:
      rw_name = self.layer.weights[1].name
    tn, ti = helper.getConstantNodeByName(rw_name, rw)
    node_list += tn
    value_list += ti
    input_list.append(rw_name)
    # Construct bias if needed
    if self.layer.cell.use_bias:
      b = self.layer.cell.get_weights()[2]
      b = np.expand_dims(w, 0)
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[2].name
      tn, ti = helper.getConstantNodeByName(bnode_name, b)
      node_list += tn
      value_list += ti
      input_list.append(bnode_name)
    # Generate Node
    output_name = self.name + "_intermediate"
    node = O.helper.make_node(
      'LSTM',
      inputs            = input_list,
      outputs           = [output_name],
      name              = self.name,
      activation_alpha  = activation_alpha,
      activation_beta   = activation_beta,
      activations       = activations,
      direction         = direction,
      hidden_size       = self.layer.cell.units
    )
    output_size = [self.node.outputs[0].keras_shape[1],
        self.node.outputs[0].keras_shape[0], 1, self.node.outputs[0].keras_shape[2]]
    output_info = O.helper.make_tensor_value_info(
        output_name,
        helper.dtype,
        output_size
    )
    node_list.append(node)
    value_list.append(output_info)
    # Construct second Transpose layer
    dims = [1, 2, 0, 3]
    postprocess_name = self.name + "_postprocess"
    postprocess_node = O.helper.make_node(
      'Transpose',
      inputs=[output_name],
      outputs=[postprocess_name],
      name=postprocess_name,
      perm=dims
    )
    postprocess_size = [1, self.node.outputs[0].shape[0],
        self.node.outputs[0].shape[1], self.node.outputs[0].shape[2]]
    postprocess_info = O.helper.make_tensor_value_info(
        postprocess_name,
        helper.dtype,
        postprocess_size
    )
    node_list.append(postprocess_node)
    value_list.append(postprocess_info)
    # Construct Reshape
    shape_name = self.name + '_shape'
    output_shape = self.output_shape
    tn, ti = helper.constructConstantNode(
      shape_name,
      np.array(output_shape, dtype='int64'))
    node_list += tn
    value_list += ti
    node = O.helper.make_node(
      op_type='Reshape',
      inputs=[postprocess_name, shape_name],
      outputs=self.outputs,
      name=self.name + "_reshape"
    )
    node_list.append(node)
    return node_list, value_list

class GRU(Layer):
  """GRU layer converter. This GRU is not bidirectional.
  A GRU layer will be converted into four layers:
  Transpose => GRU => Transpose => Reshape
  """
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Convert GRU
    node_list = []
    value_list = []
    # Keras activation layer
    keras_activation = self.layer.cell.activation.__name__
    if keras_activation not in activation_mapping:
      raise OnnxNotSupport("Activation in GRU: " + keras_activation)
    keras_activation = activation_mapping[keras_activation]
    keras_recurrent_activation = self.layer.cell.recurrent_activation.__name__
    if keras_recurrent_activation not in activation_mapping:
      raise OnnxNotSupport("Activation in GRU: " + keras_recurrent_activation)
    keras_recurrent_activation = activation_mapping[keras_recurrent_activation]
    activations = [keras_recurrent_activation, keras_activation, keras_activation]
    # Give activation alpha and beta
    activation_alpha = []
    activation_beta = []
    for activation in activations:
      if activation == 'Elu':
        activation_alpha.append(1.0)
        activation_beta.append(0.0)
      elif activation == 'HardSigmoid':
        activation_alpha.append(0.2)
        activation_beta.append(0.5)
      else:
        activation_alpha.append(0.0)
        activation_beta.append(0.0)
    # Direction
    if self.layer.go_backwards:
      direction = 'reverse'
    else:
      direction = 'forward'
    # Construct First Transpose
    input_list = []
    dims = [1, 0, 2]
    preprocess_name = self.name + "_preprocess"
    preprocess_node = O.helper.make_node(
      'Transpose',
      inputs=self.inputs,
      outputs=[preprocess_name],
      name=preprocess_name,
      perm=dims
    )
    preprocess_size = [self.node.inputs[0].keras_shape[1],
        self.node.inputs[0].keras_shape[0], self.node.inputs[0].keras_shape[2]]
    preprocess_info = O.helper.make_tensor_value_info(
        preprocess_name,
        helper.dtype,
        preprocess_size
    )
    node_list.append(preprocess_node)
    value_list.append(preprocess_info)
    input_list.append(preprocess_name)
    # Construct Weights
    w = self.layer.cell.get_weights()[0]
    w = np.transpose(w, [1, 0])
    w = np.expand_dims(w, 0)
    if helper.duplicate_weights:
      w_name = self.name + "_weight"
    else:
      w_name = self.layer.weights[0].name
    tn, ti = helper.getConstantNodeByName(w_name, w)
    node_list += tn
    value_list += ti
    input_list.append(w_name)
    # Construct recurrent weight
    rw = self.layer.cell.get_weights()[1]
    rw = np.transpose(rw, [1, 0])
    rw = np.expand_dims(rw, 0)
    if helper.duplicate_weights:
      rw_name = self.name + "_recurrent_weight"
    else:
      rw_name = self.layer.weights[1].name
    tn, ti = helper.getConstantNodeByName(rw_name, rw)
    node_list += tn
    value_list += ti
    input_list.append(rw_name)
    # Construct bias if needed
    if self.layer.cell.use_bias:
      b = self.layer.cell.get_weights()[2]
      b = np.expand_dims(w, 0)
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[2].name
      tn, ti = helper.getConstantNodeByName(bnode_name, b)
      node_list += tn
      value_list += ti
      input_list.append(bnode_name)
    # Generate Node
    output_name = self.name + "_intermediate"
    node = O.helper.make_node(
      'GRU',
      inputs            = input_list,
      outputs           = [output_name],
      name              = self.name,
      activation_alpha  = activation_alpha,
      activation_beta   = activation_beta,
      activations       = activations,
      direction         = direction,
      hidden_size       = self.layer.cell.units
    )
    output_size = [self.node.outputs[0].keras_shape[1],
        self.node.outputs[0].keras_shape[0], 1, self.node.outputs[0].keras_shape[2]]
    output_info = O.helper.make_tensor_value_info(
        output_name,
        helper.dtype,
        output_size
    )
    node_list.append(node)
    value_list.append(output_info)
    # Construct second Transpose layer
    dims = [1, 2, 0, 3]
    postprocess_name = self.name + "_postprocess"
    postprocess_node = O.helper.make_node(
      'Transpose',
      inputs=[output_name],
      outputs=[postprocess_name],
      name=postprocess_name,
      perm=dims
    )
    postprocess_size = [1, self.node.outputs[0].shape[0],
        self.node.outputs[0].shape[1], self.node.outputs[0].shape[2]]
    postprocess_info = O.helper.make_tensor_value_info(
        postprocess_name,
        helper.dtype,
        postprocess_size
    )
    node_list.append(postprocess_node)
    value_list.append(postprocess_info)
    # Construct Reshape
    shape_name = self.name + '_shape'
    output_shape = self.output_shape
    tn, ti = helper.constructConstantNode(
      shape_name,
      np.array(output_shape, dtype='int64'))
    node_list += tn
    value_list += ti
    node = O.helper.make_node(
      op_type='Reshape',
      inputs=[postprocess_name, shape_name],
      outputs=self.outputs,
      name=self.name + "_reshape"
    )
    node_list.append(node)
    return node_list, value_list
