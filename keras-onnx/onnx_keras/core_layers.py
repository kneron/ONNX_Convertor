"""Converters for core layers in Keras
"""
import onnx as O
import keras as K
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport, NoneStandardKerasModel
from . import helper

class Flatten(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node = O.helper.make_node(
        op_type='Flatten',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        axis=1
    )
    return [node], []
  @staticmethod
  def isFlatten(node):
    '''Check if a tree node is a node function as Flatten.

    param node: TreeNode
    return: True or False
    '''
    if node.type == 'Flatten':
      return True
    elif node.type == 'Reshape':
      if len(node.klayer.target_shape) == 1:
        return True
      else:
        return False
    else:
      return False

class Activation(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    act = K.activations.serialize(self.layer.activation)
    if act == 'relu':
      return self.relu()
    elif act == 'relu6':
      helper.logger.error("Please update your keras model. Use relu with max attribute instead of relu6.")
      raise NoneStandardKerasModel("Custom relu6")
    elif act == 'softmax':
      return self.softmax()
    elif act == 'elu':
      return self.elu()
    elif act == 'linear':
      return self.linear()
    elif act == 'sigmoid':
      return self.sigmoid()
    elif act == 'tanh':
      return self.tanh()
    elif act == 'hard_sigmoid':
      return self.hard_sigmoid()
    else:
      #raise FeatureNotImplemented('OP Activation ' + act)
      helper.logger.warning('Activation ' + act + ' is currently not supported. Treated as mystery layer')
      return self.mystery(act)

  # Special activation operation
  def mystery(self, target):
    node = O.helper.make_node(
      op_type='Mystery',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      target=target
    )
    return [node], []

  # Activation operations
  def sigmoid(self):
    node = O.helper.make_node(
      op_type='Sigmoid',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
    )
    return [node], []

  def softmax(self, axis=-1):
    transpose_node_list = []

    # 4 dim input has channel last-first op definition issue, need transpose
    do_transpose = len(self.input_shapes[0]) is 4

    if do_transpose is True:
      transpose_before_node_name = 'transpose_node_before_' + self.name
      transpose_before_node = O.helper.make_node(
          'Transpose',
          inputs=self.inputs,
          outputs=[transpose_before_node_name],
          perm=[0,2,3,1],
          name=transpose_before_node_name
      )

      transpose_after_node_name = 'transpose_node_after_' + self.name
      transpose_after_node = O.helper.make_node(
          'Transpose',
          inputs=[self.name],
          outputs=self.outputs,
          perm=[0,3,1,2],
          name=transpose_after_node_name
      )

      transpose_node_list = [transpose_before_node, transpose_after_node]
    else:
      # change axis param directly
      mapping_table = {'0':0, '3':1, '1':2, '2':3, '-1': 1}
      axis = mapping_table[str(axis)]

    node = O.helper.make_node(
      op_type='Softmax',
      inputs=self.inputs if do_transpose is False else [transpose_node_list[0].name],
      outputs=self.outputs if do_transpose is False else [self.name],
      name=self.name,
      axis=axis
    )

    if do_transpose is True:
      return [transpose_node_list[0], node, transpose_node_list[1]], []
    else:
      return [node], []

  def relu(self):
    node = O.helper.make_node(
      op_type='Relu',
      inputs = self.inputs,
      outputs = self.outputs,
      name=self.name
    )
    return [node], []

  def elu(self, alpha=1.0):
    node = O.helper.make_node(
      op_type='Elu',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      alpha=alpha
    )
    return [node], []

  def linear(self):
    node = O.helper.make_node(
      op_type='Identity',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
    )
    return [node], []

  def tanh(self):
    node = O.helper.make_node(
      op_type='Tanh',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
    )
    return [node], []

  def hard_sigmoid(self):
    node = O.helper.make_node(
      op_type = 'HardSigmoid',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name,
      alpha = 0.2,
      beta = 0.5
    )
    return [node], []

class Dense(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def checkSecondFC(self):
    second_fc = False
    node = self.node.inputs[0].input
    skip_layers = [
      'Activation',
      'ReLU',
      'LeakyReLU',
      'PReLU',
      'Softmax',
      'Dropout'
    ]
    while node is not None:
      if node.type == 'Dense':
        second_fc = True
        break
      elif node.type == 'InputLayer':
        second_fc = False
        break
      elif node.type in skip_layers:
        node = node.inputs[0].input
        continue
      else:
        second_fc = False
        break
    return second_fc
  def get_channel_number(self):
    skip = ['Flatten', 'Dense', 'Dropout']
    node = self.node.inputs[0].input
    while node is not None:
      if node.type in skip or Flatten.isFlatten(node):
        node = node.inputs[0].input
      else:
        return node.outputs[0].keras_shape[-1]
    raise ValueError("Dense layer cannot find the channel number for reshape.")
  def generate(self):
    # Prepare variables
    inputs = self.inputs
    node_list = []
    value_infos = []
    # Replace Dence(FC) with GEMM
    self.logger.info("Dense is treated as Gemm")
    # Construct Weights
    # Need reshape from channels_last to channels_first
    if helper.duplicate_weights:
      wnode_name = self.name + "_weight"
    else:
      wnode_name = self.layer.weights[0].name
    w_data = self.layer.get_weights()[0]

    # If wnode is known, no need to construct it again.
    if wnode_name not in helper.known_tensors:
      # Check if this dense is following another dense
      second_fc = self.checkSecondFC()
      if helper.data_format == 'channels_last' and not second_fc:
        helper.logger.info("{} is not the second Dense. Weight reshaping is needed.".format(self.name))
        w_data = w_data.reshape([-1, self.get_channel_number(), w_data.shape[1]])
        w_data = w_data.transpose([1, 0, 2])
        w_data = w_data.reshape([-1, w_data.shape[2]])

    tn, ti = helper.getConstantNodeByName(wnode_name, w_data)
    node_list += tn
    value_infos += ti
    inputs.append(wnode_name)
    # Construct Bias
    if not self.layer.use_bias:
      b_data = np.zeros(w_data.shape[1], dtype=w_data.dtype)
      bnode_name = self.name + "_bias"
    else:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[1].name
      b_data = self.layer.get_weights()[1]
    tn, ti = helper.getConstantNodeByName(
      bnode_name,
      b_data)
    node_list += tn
    value_infos += ti
    inputs.append(bnode_name)
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      final_out = self.outputs
      self.outputs = [final_out[0][:-2] + 'inner_act']
    else:
      final_out = None
    # Make the node
    node = O.helper.make_node(
      op_type   = 'Gemm',
      inputs    = inputs,
      outputs   = self.outputs,
      name      = self.name,
      alpha     = 1.0,
      beta      = 1.0,
      transA    = 0,
      transB    = 0
      )
    node_list.append(node)
    # Make the activation node
    if final_out is not None:
      act = Activation(self.node)
      act.inputs = self.outputs
      act.outputs = final_out
      act.name = self.name + "_activation"
      act_nodes, act_values = act.generate()
      act_info = O.helper.make_tensor_value_info(
        self.outputs[0],
        helper.dtype,
        self.node.outputs[0].shape
      )
      node_list += act_nodes
      value_infos += act_values
      value_infos.append(act_info)
      self.outputs = final_out
    return node_list, value_infos

class Reshape(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    helper.logger.info("Critical layer: Reshape to {} (target: {})".format(str(self.output_shape), self.node.klayer.target_shape))
    if Flatten.isFlatten(self.node):
      helper.logger.info("Treating Reshape {} as Flatten".format(self.name))
      new_generator = Flatten(self.node)
      return new_generator.generate()
    node_list = []
    value_list = []
    inputs = self.inputs
    outputs = self.outputs
    output_shape = self.output_shape
    # Check if the channel remains the same before and after Reshape.
    # If different, construct Transpose layer before and after Reshape.
    # But for RNN, no need to tranpose back
    need_transpose = (self.input_shapes[0][1] != self.output_shape[1] and not helper.RNN_start)
    # Check if this is the start node of RNN
    if helper.RNN_start_node == self.node:
      helper.RNN_start = True
      helper.logger.debug("RNN starts")
    # Construct first transpose
    if need_transpose:
      transpose_name = self.name + '_transpose0'
      inputs = [transpose_name]
      if not helper.RNN_start:
        # if RNN start node, no need for second transpose
        outputs = [self.name + '_inner']
      output_shape = self.node.outputs[0].keras_shape
      dims = list(range(len(self.input_shapes[0])))
      dims = dims[:1] + dims[2:] + dims[1:2]
      transpose_node = O.helper.make_node(
        'Transpose',
        inputs=self.inputs,
        outputs=inputs,
        name=transpose_name,
        perm=dims
      )
      transpose_value = O.helper.make_tensor_value_info(
        transpose_name,
        helper.dtype,
        self.node.inputs[0].keras_shape
      )
      node_list.append(transpose_node)
      value_list.append(transpose_value)
    # Construct shape
    shape_name = self.name + '_shape'
    tn, ti = helper.constructConstantNode(
      shape_name,
      np.array(output_shape, dtype='int64'))
    node_list += tn
    value_list += ti
    inputs.append(shape_name)
    # Construct node
    node = O.helper.make_node(
      op_type='Reshape',
      inputs=inputs,
      outputs=outputs,
      name=self.name
    )
    node_list.append(node)
    # Construct second transpose
    if need_transpose and not helper.RNN_start:
      transpose_name = self.name + '_transpose1'
      transpose_value = O.helper.make_tensor_value_info(
        outputs[0],
        helper.dtype,
        output_shape
      )
      dims = list(range(len(self.output_shape)))
      dims = dims[:1] + dims[-1:] + dims[1:-1]
      transpose_node = O.helper.make_node(
        'Transpose',
        inputs=outputs,
        outputs=self.outputs,
        name=transpose_name,
        perm=dims
      )
      node_list.append(transpose_node)
      value_list.append(transpose_value)
    # Return
    return node_list, value_list

class Dropout(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node = O.helper.make_node(
      'Dropout',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      ratio=self.layer.rate
    )
    return [node], []

class Permute(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    dims = list(self.layer.dims)
    dims.insert(0, 0)
    node = O.helper.make_node(
      'Transpose',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      perm=dims
    )
    return [node], []
