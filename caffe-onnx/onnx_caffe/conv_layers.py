"""Converters for convolution layers in Keras
"""
import onnx as O
import numpy as np

from . import helper
from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport

class Convolution(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    value_infos = []
    # Construct Weights.
    w = self.layer.blobs[0].data
    wnode_name = self.name + '_weight'
    tn, ti = helper.constructConstantNode(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.proto.convolution_param.bias_term == True:
      bnode_name = self.name + '_bias'
      tn, ti = helper.constructConstantNode(
        bnode_name, self.layer.blobs[1].data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)

    # Construct other params
    if self.proto.convolution_param.kernel_w > 0 and self.proto.convolution_param.kernel_h > 0:
      kernel = [self.proto.convolution_param.kernel_h, self.proto.convolution_param.kernel_w]
    else:
      if len(self.proto.convolution_param.kernel_size):
        kernel = self.proto.convolution_param.kernel_size[0]
      else:
        kernel = 1
      kernel = [kernel] * 2

    if self.proto.convolution_param.pad_w > 0 or self.proto.convolution_param.pad_h > 0:
      pad = [self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w, self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w]
    else:
      if len(self.proto.convolution_param.pad):
        pad = self.proto.convolution_param.pad[0]
      else:
        pad = 0
      pad = [pad] * 4

    if self.proto.convolution_param.stride_w > 0 or self.proto.convolution_param.stride_h > 0:
      stride = [self.proto.convolution_param.stride_h, self.proto.convolution_param.stride_w]
    else:
      if len(self.proto.convolution_param.stride):
        stride = self.proto.convolution_param.stride[0]
      else:
        stride = 1
      stride = [stride] * 2

    dilation = self.proto.convolution_param.dilation[0] if len(self.proto.convolution_param.dilation) else 1
    dilation = [dilation] * 2
    group    = self.proto.convolution_param.group
    node = O.helper.make_node(
      op_type       = 'Conv',
      inputs        = self.inputs,
      outputs       = self.outputs,
      name          = self.name,
      dilations     = dilation,
      kernel_shape  = kernel,
      pads          = pad,
      strides       = stride,
      group         = group
      )
    node_list.append(node)
    return node_list, value_infos

class DepthwiseConvolution(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    value_infos = []
    # Construct Weights.
    w = self.layer.blobs[0].data
    wnode_name = self.name + '_weight'
    tn, ti = helper.constructConstantNode(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.proto.convolution_param.bias_term == True:
      bnode_name = self.name + '_bias'
      tn, ti = helper.constructConstantNode(
        bnode_name, self.layer.blobs[1].data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)

    # Construct other params
    if self.proto.convolution_param.kernel_w > 0 and self.proto.convolution_param.kernel_h > 0:
      kernel = [self.proto.convolution_param.kernel_h, self.proto.convolution_param.kernel_w]
    else:
      if len(self.proto.convolution_param.kernel_size):
        kernel = self.proto.convolution_param.kernel_size[0]
      else:
        kernel = 1
      kernel = [kernel] * 2

    if self.proto.convolution_param.pad_w > 0 or self.proto.convolution_param.pad_h > 0:
      pad = [self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w, self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w]
    else:
      if len(self.proto.convolution_param.pad):
        pad = self.proto.convolution_param.pad[0]
      else:
        pad = 0
      pad = [pad] * 4

    if self.proto.convolution_param.stride_w > 0 or self.proto.convolution_param.stride_h > 0:
      stride = [self.proto.convolution_param.stride_h, self.proto.convolution_param.stride_w]
    else:
      if len(self.proto.convolution_param.stride):
        stride = self.proto.convolution_param.stride[0]
      else:
        stride = 1
      stride = [stride] * 2

    dilation = self.proto.convolution_param.dilation[0] if len(self.proto.convolution_param.dilation) else 1
    dilation = [dilation] * 2
    group    = self.proto.convolution_param.group
    node = O.helper.make_node(
      op_type       = 'Conv',
      inputs        = self.inputs,
      outputs       = self.outputs,
      name          = self.name,
      dilations     = dilation,
      kernel_shape  = kernel,
      pads          = pad,
      strides       = stride,
      group         = group
      )
    node_list.append(node)
    return node_list, value_infos

class Deconvolution(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    value_infos = []
    # Construct Weights.
    w = self.layer.blobs[0].data
    wnode_name = self.name + '_weight'
    tn, ti = helper.constructConstantNode(wnode_name, w)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.proto.convolution_param.bias_term == True:
      bnode_name = self.name + '_bias'
      tn, ti = helper.constructConstantNode(
        bnode_name, self.layer.blobs[1].data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)

    # Construct other params
    if self.proto.convolution_param.kernel_w > 0 and self.proto.convolution_param.kernel_h > 0:
      kernel = [self.proto.convolution_param.kernel_h, self.proto.convolution_param.kernel_w]
    else:
      if len(self.proto.convolution_param.kernel_size):
        kernel = self.proto.convolution_param.kernel_size[0]
      else:
        kernel = 1
      kernel = [kernel] * 2

    if self.proto.convolution_param.pad_w > 0 or self.proto.convolution_param.pad_h > 0:
      pad = [self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w, self.proto.convolution_param.pad_h, self.proto.convolution_param.pad_w]
    else:
      if len(self.proto.convolution_param.pad):
        pad = self.proto.convolution_param.pad[0]
      else:
        pad = 0
      pad = [pad] * 4

    if self.proto.convolution_param.stride_w > 0 or self.proto.convolution_param.stride_h > 0:
      stride = [self.proto.convolution_param.stride_h, self.proto.convolution_param.stride_w]
    else:
      if len(self.proto.convolution_param.stride):
        stride = self.proto.convolution_param.stride[0]
      else:
        stride = 1
      stride = [stride] * 2

    dilation = self.proto.convolution_param.dilation[0] if len(self.proto.convolution_param.dilation) else 1
    dilation = [dilation] * 2
    group    = self.proto.convolution_param.group
    node = O.helper.make_node(
      op_type       = 'ConvTranspose',
      inputs        = self.inputs,
      outputs       = self.outputs,
      name          = self.name,
      dilations     = dilation,
      kernel_shape  = kernel,
      pads          = pad,
      strides       = stride,
      group         = group
      )
    node_list.append(node)
    return node_list, value_infos
