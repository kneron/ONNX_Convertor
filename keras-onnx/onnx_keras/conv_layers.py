"""Converters for convolution layers in Keras
"""
import onnx as O
import numpy as np
import copy

from . import helper
from .base_layer import Layer
from .core_layers import Activation
from .exceptions import FeatureNotImplemented, OnnxNotSupport

class Conv2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node_list = []
    value_infos = []
    # Construct Weights. Keras format(HWCM). M output channel. C input channel.
    if self.node.new_w is not None:
      wnode_name = self.name + '_fused_weight'
      w = self.node.new_w
    else:
      if helper.duplicate_weights:
        wnode_name = self.name + "_weight"
      else:
        wnode_name = self.layer.weights[0].name
      w = self.layer.get_weights()[0]
    if wnode_name not in helper.known_tensors and self.node.new_w is None:
      w = np.transpose(w, [3,2,0,1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.node.new_b is not None:
      bnode_name = self.name + '_fused_bias'
      tn, ti = helper.getConstantNodeByName(
        bnode_name, self.node.new_b)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    elif self.layer.use_bias:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[1].name
      tn, ti = helper.getConstantNodeByName(
        bnode_name, self.layer.get_weights()[1])
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    elif helper.compatibility:
      bnode_name = self.name + '_constructed_bias'
      constructed_data = np.zeros(w.shape[0], dtype=w.dtype)
      tn, ti = helper.getConstantNodeByName(
        bnode_name, constructed_data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    # Calculate actual kernel shape
    kernel_size = list(self.layer.kernel_size)
    dilations = list(self.layer.dilation_rate)
    kernel_size[0] = (kernel_size[0] - 1) * dilations[0] + 1
    kernel_size[1] = (kernel_size[1] - 1) * dilations[1] + 1
    # Construct padding_array
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
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      final_out = self.outputs
      self.outputs = [final_out[0][:-2] + 'inner_act']
    else:
      final_out = None
    # Make the node
    node = O.helper.make_node(
      op_type='Conv',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      dilations=list(self.layer.dilation_rate),
      kernel_shape=list(self.layer.kernel_size),
      pads=padding_array,
      strides=list(self.layer.strides)
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

class Conv2DTranspose(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node_list = []
    value_infos = []
    # The weight and bias are processed similarly as the Conv2D
    # Construct Weights. Keras format(HWCM)
    if helper.duplicate_weights:
      wnode_name = self.name + "_weight"
    else:
      wnode_name = self.layer.weights[0].name
    w = self.layer.get_weights()[0]
    if wnode_name not in helper.known_tensors:
      w = np.transpose(w, [3,2,0,1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.layer.use_bias:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[1].name
      tn, ti = helper.getConstantNodeByName(
        bnode_name, self.layer.get_weights()[1])
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    # Attributes that can be applied directly
    dilations = list(self.layer.dilation_rate)
    kernel_shape = list(self.layer.kernel_size)
    strides = list(self.layer.strides)
    # Check padding
    if self.layer.padding == 'same':
        auto_pad = 'SAME_LOWER'
    else:
        auto_pad = 'VALID'
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      final_out = self.outputs
      self.outputs = [final_out[0][:-2] + 'inner_act']
    else:
      final_out = None
    # Make the node
    if self.layer.output_padding is None:
        node = O.helper.make_node(
            op_type='ConvTranspose',
            inputs=self.inputs,
            outputs=self.outputs,
            name=self.name,
            dilations=dilations,
            kernel_shape=kernel_shape,
            auto_pad=auto_pad,
            output_shape=self.output_shape[2:],
            strides=strides
        )
    else:
        node = O.helper.make_node(
            op_type='ConvTranspose',
            inputs=self.inputs,
            outputs=self.outputs,
            name=self.name,
            dilations=dilations,
            kernel_shape=kernel_shape,
            auto_pad=auto_pad,
            strides=strides,
            pads=[int(self.layer.output_padding[0])//2,
                            int(self.layer.output_padding[1])//2,
                            int(self.layer.output_padding[0])//2 + int(self.layer.output_padding[0])%2,
                            int(self.layer.output_padding[1])//2 + int(self.layer.output_padding[1])%2]
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

class ZeroPadding2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Construct padding array (None 2D dimension pad 0)
    pads = []
    for _ in range(len(self.layer.output_shape) - 2):
      pads.append(0)
    pads += [self.layer.padding[0][0], self.layer.padding[1][0]]
    for _ in range(len(self.layer.output_shape) - 2):
      pads.append(0)
    pads += [self.layer.padding[0][1], self.layer.padding[1][1]]
    # Make the node
    node = O.helper.make_node(
      op_type='Pad',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      pads=pads
      )
    return [node], []

class DepthwiseConv2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)

  def generate_bn(self):
    node_list = []
    value_infos = []
    # Construct Scale
    if helper.duplicate_weights:
      wnode_name = self.name + "_scale"
    else:
      wnode_name = self.layer.weights[0].name
    w = self.layer.get_weights()[0]
    w = np.reshape(w, [-1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.layer.use_bias:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[1].name
      tn, ti = helper.getConstantNodeByName(bnode_name, self.layer.get_weights()[1])
    else:
      bnode_name = self.name + '_constructed_bias'
      constructed_data = np.zeros(w.shape[0], dtype=w.dtype)
      tn, ti = helper.getConstantNodeByName(bnode_name, constructed_data)
    node_list += tn
    value_infos += ti
    self.inputs.append(bnode_name)
    # Construct Mean
    mean_name = self.name + '_constructed_mean'
    constructed_data = np.zeros(w.shape[0], dtype=w.dtype)
    tn, ti = helper.getConstantNodeByName(mean_name, constructed_data)
    node_list += tn
    value_infos += ti
    self.inputs.append(mean_name)
    # Construct Var
    var_name = self.name + '_constructed_var'
    constructed_data = np.ones(w.shape[0], dtype=w.dtype)
    tn, ti = helper.getConstantNodeByName(var_name, constructed_data)
    node_list += tn
    value_infos += ti
    self.inputs.append(var_name)
    # Construct layer
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      final_out = self.outputs
      self.outputs = [final_out[0][:-2] + 'inner_act']
    else:
      final_out = None
    # Make the node
    node = O.helper.make_node(
      op_type='BatchNormalization',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
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

  def generate(self):
    node_list = []
    value_infos = []
    # Calculate actual kernel shape
    kernel_size = list(self.layer.kernel_size)
    dilations = list(self.layer.dilation_rate)
    kernel_size[0] = (kernel_size[0] - 1) * dilations[0] + 1
    kernel_size[1] = (kernel_size[1] - 1) * dilations[1] + 1
    if kernel_size[0] == 1 and kernel_size[1] == 1 and self.layer.depth_multiplier == 1:
      return self.generate_bn()
    # Construct Weights. Keras format(HWCM)
    if helper.duplicate_weights:
      wnode_name = self.name + "_weight"
    else:
      wnode_name = self.layer.weights[0].name
    w = self.layer.get_weights()[0]
    if wnode_name not in helper.known_tensors:
      if (self.layer.depth_multiplier != 1):
        w = np.reshape(w, [self.layer.kernel_size[0], self.layer.kernel_size[1], -1, 1])
      w = np.transpose(w, [2,3,0,1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.layer.use_bias:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[1].name
      tn, ti = helper.getConstantNodeByName(
        bnode_name, self.layer.get_weights()[1])
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    # Construct padding_array
    if self.layer.padding == 'valid':
      padding_array = [0, 0, 0, 0]
    else:
      input_shape = self.input_shapes[0]
      padding_array = helper.getPadding(
        input_shape[2:4],
        kernel_size,
        self.layer.strides)
    # Get channel number
    if helper.data_format == 'channels_last':
      channel = self.layer.input_shape[3]
    else:
      channel = self.layer.input_shape[1]
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      final_out = self.outputs
      self.outputs = [final_out[0][:-2] + 'inner_act']
    else:
      final_out = None
    # Make the node
    node = O.helper.make_node(
      op_type='Conv',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      dilations=list(self.layer.dilation_rate),
      kernel_shape=list(self.layer.kernel_size),
      pads=padding_array,
      strides=list(self.layer.strides),
      group=channel
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

class UpSampling2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    nodes = []
    values = []
    # Construct scales
    scales = list(self.layer.size)
    for _ in range(len(self.layer.input_shape) - len(scales)):
      scales.insert(0, 1)
    if not helper.compatibility:
      scale_name = self.name + '_scales'
      tn, ti = helper.constructConstantNode(
        scale_name,
        np.array(scales, dtype='float32'))
      nodes += tn
      values += ti
      self.inputs.append(scale_name)
    # Setup shape
    if self.layer.interpolation is not None:
      if self.layer.interpolation == 'nearest':
        mode = 'nearest'
      else:
        mode = 'linear'
    else:
      # Default is nearest
      mode = 'nearest'

    # Make the node
    if helper.compatibility:
      node = O.helper.make_node(
        op_type='Upsample',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        mode=mode,
        scales=list(map(float, scales))
        )
    else:
      node = O.helper.make_node(
        op_type='Upsample',
        inputs=self.inputs,
        outputs=self.outputs,
        name=self.name,
        mode=mode
        )
    nodes.append(node)
    return nodes, values

class Cropping2D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Construct
    input_shape = self.input_shapes[0]
    if helper.data_format == 'channel_last':
      crop_shape = input_shape[1:3]
    else:
      crop_shape = input_shape[2:4]
    crop = self.layer.cropping
    if isinstance(crop[0], int):
      crop = ((crop[0], crop[0]), (crop[1], crop[1]))
    slice_starts = [crop[0][0], crop[1][0]]
    slice_ends = [crop_shape[0] - crop[0][1], crop_shape[1] - crop[1][1]]
    node = O.helper.make_node(
      op_type = 'Slice',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      axes=[2, 3],
      ends=slice_ends,
      starts=slice_starts
    )
    return [node], []

class Cropping1D(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    # Construct
    crop_shape = self.input_shapes[0]
    crop = self.layer.cropping[0]
    slice_starts = [crop[0]]
    slice_ends = [crop_shape[-1] - crop[1]]
    node = O.helper.make_node(
      op_type = 'Slice',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      axes=[len(crop_shape) - 1],
      ends=slice_ends,
      starts=slice_starts
    )
    return [node], []

class SeparableConv2D(Layer):
  """Separable conv consists of a depthwise    # Construct Weights. Keras format(HWCM)
 conv and a pointwise conv
  """
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node_list = []
    value_infos = []
    # 1. Construct Depthwise Conv
    depthwise_input = self.inputs
    # Construct Weights. Keras format(HWCM)
    if helper.duplicate_weights:
      wnode_name = self.name + "_weight"
    else:
      wnode_name = self.layer.weights[0].name
    w = self.layer.get_weights()[0]
    if wnode_name not in helper.known_tensors:
      if (self.layer.depth_multiplier != 1):
        w = np.reshape(w, [self.layer.kernel_size[0], self.layer.kernel_size[1], -1, 1])
      w = np.transpose(w, [2,3,0,1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    depthwise_input.append(wnode_name)
    # Construct padding_array
    if self.layer.padding == 'valid':
      padding_array = [0, 0, 0, 0]
    else:
      input_shape = self.input_shapes[0]
      padding_array = helper.getPadding(
        input_shape[2:4],
        self.layer.kernel_size,
        self.layer.strides)
    # Get channel number
    if helper.data_format == 'channels_last':
      channel = self.layer.input_shape[3]
    else:
      channel = self.layer.input_shape[1]
    # Make the node
    conv_input = [self.name + '_depthwise_output']
    node = O.helper.make_node(
      op_type       = 'Conv',
      inputs        = depthwise_input,
      outputs       = conv_input,
      name          = self.name + '_depthwise',
      dilations     = list(self.layer.dilation_rate),
      kernel_shape  = list(self.layer.kernel_size),
      pads          = padding_array,
      strides       = list(self.layer.strides),
      group         = channel
      )
    node_list.append(node)
    # Construct depthwise conv output value info
    dw_output_shape = [self.input_shapes[0][0],
        self.input_shapes[0][1] * self.layer.depth_multiplier,
        self.node.outputs[0].shape[2], self.node.outputs[0].shape[3]]
    dw_info = O.helper.make_tensor_value_info(
      conv_input[0],
      helper.dtype,
      dw_output_shape
    )
    value_infos.append(dw_info)

    # 2. Construct Conv
    # Construct Weights. Keras format(HWCM)
    if helper.duplicate_weights:
      wnode_name = self.name + "_weight"
    else:
      wnode_name = self.layer.weights[1].name
    w = self.layer.get_weights()[1]
    if wnode_name not in helper.known_tensors:
      w = np.transpose(w, [3,2,0,1])
    tn, ti = helper.getConstantNodeByName(wnode_name, w)
    node_list += tn
    value_infos += ti
    conv_input.append(wnode_name)
    # Construct Bias
    if self.layer.use_bias:
      if helper.duplicate_weights:
        bnode_name = self.name + "_bias"
      else:
        bnode_name = self.layer.weights[2].name
      tn, ti = helper.getConstantNodeByName(
        bnode_name, self.layer.get_weights()[2])
      node_list += tn
      value_infos += ti
      conv_input.append(bnode_name)
    # Construct pad1, 1ding_array
    padding_array = [0, 0, 0, 0]
    # If the layer has activation, the convolution should have an middle output.
    if helper.hasActivation(self.layer.activation):
      conv_output = [self.name + '_conv_output']
    else:
      conv_output = self.outputs
    # Make the node
    node = O.helper.make_node(
      op_type       = 'Conv',
      inputs        = conv_input,
      outputs       = conv_output,
      name          = self.name + '_conv',
      dilations     = [1, 1],
      kernel_shape  = [1, 1],
      pads          = padding_array,
      strides       = [1, 1]
      )
    node_list.append(node)

    # 3. Construct Activation
    if helper.hasActivation(self.layer.activation):
      # Make intermedia value info
      conv_info = O.helper.make_tensor_value_info(
        conv_output[0],
        helper.dtype,
        self.node.outputs[0].shape
      )
      value_infos.append(conv_info)
      # Make activation node
      act = Activation(self.node)
      act.inputs = conv_output
      act.outputs = self.outputs
      act.name = self.name + "_activation"
      act_nodes, act_values = act.generate()
      node_list += act_nodes
      value_infos += act_values
    return node_list, value_infos
