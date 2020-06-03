"""Converters for pooling layers in Keras
"""
import onnx as O

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper
'''
eltwise_param {
operation: SUM
}
'''

class Eltwise(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    #print(self.inputs, self.outputs)
    if self.proto.eltwise_param.operation == 0:
      op_type = 'Mul'
    elif self.proto.eltwise_param.operation == 1:
      op_type = 'Add'
    elif self.proto.eltwise_param.operation == 2:
      op_type = 'Max'
    node = O.helper.make_node(
      op_type = op_type,
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name)
    return [node], []


class Concat(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    #print(self.proto)
    axis = 1
    if (self.proto.concat_param):
      if self.proto.concat_param.axis:
        axis = self.proto.concat_param.axis
    node = O.helper.make_node(
      op_type = 'Concat',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name,
      axis = axis
    )
    return [node], []