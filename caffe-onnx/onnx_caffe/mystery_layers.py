"""Converters for unknown layers and layers without Onnx support in Caffe.
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class Mystery(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
    if self.name in helper.custom_name2type:
      helper.logger.debug("Found type classification for lambda layer " + self.name)
      self.type = helper.custom_name2type[self.name]
      self.opid = helper.custom_type2opid[self.type]
    else:
      helper.logger.warning("Lambda layer " + self.name + " is not properly defined in custom.json. Converting to Mystery layer.")
      helper.unknown_types.add(self.layer.type)
      self.opid = helper.opid_counter
      helper.opid_counter += 1
      self.type = "CustomOP" + str(self.opid)
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

class Python(Mystery):
  def __init__(self, inputs, outname, layer, proto, blob):
    Mystery.__init__(self, inputs, outname, layer, proto, blob)

