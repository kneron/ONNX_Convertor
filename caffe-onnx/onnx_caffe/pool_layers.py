"""Converters for pooling layers in Keras
"""
import onnx as O

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class Pooling(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    stride = self.proto.pooling_param.stride if self.proto.pooling_param.stride else 1
    pad    = self.proto.pooling_param.pad if self.proto.pooling_param.pad else 0
    if self.proto.pooling_param.kernel_size:
      kernel = [self.proto.pooling_param.kernel_size] * 2
    elif self.proto.pooling_param.kernel_h:
      kernel = [self.proto.pooling_param.kernel_h, self.proto.pooling_param.kernel_w]
    else:
      kernel = [1, 1]
    stride = [stride] * 2
    pads  = helper.getPadding([self.blob.data.shape[2], self.blob.data.shape[3]], kernel, stride, pad)

    node = []
    if self.proto.pooling_param.global_pooling == True:
      if (self.proto.pooling_param.pool == 0):
        node = O.helper.make_node(
          op_type='GlobalMaxPool',
          inputs=self.inputs,
          outputs=self.outputs,
          name=self.name)
      else:
        node = O.helper.make_node(
          op_type='GlobalAveragePool',
          inputs=self.inputs,
          outputs=self.outputs,
          name=self.name)
    else:
      if (self.proto.pooling_param.pool == 0):
        node = O.helper.make_node(
          op_type = 'MaxPool',
          inputs = self.inputs,
          outputs = self.outputs,
          name = self.name,
          kernel_shape = kernel,
          pads = pads,
          strides = stride
        )
      else:
        node = O.helper.make_node(
          op_type = 'AveragePool',
          inputs = self.inputs,
          outputs = self.outputs,
          name = self.name,
          kernel_shape = kernel,
          pads = pads,
          strides = stride
        )
    return [node], []

class ROIPooling(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    pooled_shape = [0, 0]
    if self.proto.roi_pooling_param.pooled_h:
      pooled_shape[0] = self.proto.roi_pooling_param.pooled_h
    if self.proto.roi_pooling_param.pooled_w:
      pooled_shape[1] = self.proto.roi_pooling_param.pooled_w
    if self.proto.roi_pooling_param.spatial_scale:
      spatial_scale = self.proto.roi_pooling_param.spatial_scale
    else:
      spatial_scale = 1
    node = O.helper.make_node(
      op_type='MaxRoiPool',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      pooled_shape=pooled_shape,
      spatial_scale=spatial_scale)
    return [node], []
