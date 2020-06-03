"""Converters for core layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper


class InnerProduct(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    value_infos = []
    # Replace Dence(FC) with GEMM
    self.logger.info("InnerProduct is treated as Gemm")
    node_0 = O.helper.make_node(
        op_type='Flatten',
        inputs=self.inputs,
        outputs=[self.name+'_0_out'],
        name=self.name+'_0',
        axis=1
      )
    node_list.append(node_0)
    self.inputs = [self.name+'_0_out']
    # Check activation
    #if helper.hasActivation(self.layer.activation):
      #raise FeatureNotImplemented("Activation {} inside {}".format(layer.activation.__name__, str(layer.name)))
      #self.logger.warning("Activation {} inside {} is ignored.".format(self.layer.activation.__name__, str(self.name)))
    # Construct Weights
    # Need reshape from channels_last to channels_first
    #print(self.proto)#.convolution_param.bias_term == True:)
    w_data = self.layer.blobs[0].data.transpose()
    #print(w_data.shape)
    #print(self.layer.blobs[1].data.shape)
    info = O.helper.make_tensor_value_info(
      self.name+'_0_out',
      helper.convertKerasType(helper.dtype),
      [self.blob.data.shape[0], w_data.shape[0]]
    )
    value_infos.append(info)
    wnode_name = self.name + '_weight'
    tn, ti = helper.constructConstantNode(wnode_name, w_data)
    node_list += tn
    value_infos += ti
    self.inputs.append(wnode_name)
    # Construct Bias
    if self.proto.inner_product_param.bias_term:
      # raise OnnxNotSupport("Dense without bias")
      bnode_name = self.name + '_bias'
      tn, ti = helper.constructConstantNode(
        bnode_name,
        self.layer.blobs[1].data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    else:
      bnode_name = self.name + "_bias"
      data = np.zeros(self.layer.blobs[0].shape[0], dtype=np.float32)
      tn, ti = helper.constructConstantNode(
        bnode_name,
        data)
      node_list += tn
      value_infos += ti
      self.inputs.append(bnode_name)
    node_1 = O.helper.make_node(
      op_type = 'Gemm',
      inputs = self.inputs,
      outputs = self.outputs,
      name = self.name+'_1',
    #  alpha = 1.0,
    #  beta = 1.0,
    #  broadcast = 0,
    #  transA = 0,
    #  transB = 0
      )
    node_list.append(node_1)
    return node_list, value_infos

class Dropout(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node = O.helper.make_node(
      'Dropout',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      ratio=self.proto.dropout_param.dropout_ratio
    )
    return [node], []

class Reshape(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    #print(self.proto)
    shape_name = self.name + '_shape'
    #print(self.proto.reshape_param.shape)
    #print(self.proto.reshape_param.shape.dim)
    output_shape = self.proto.reshape_param.shape.dim
    #print(output_shape)
    tn, ti = helper.constructConstantNode(
      shape_name,
      np.array(output_shape, dtype='int64'))
    node_list += tn
    self.inputs.append(shape_name)
    node = O.helper.make_node(
      op_type='Reshape',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name
    )
    node_list.append(node)
    return node_list, ti

class Flatten(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node = O.helper.make_node(
      op_type='Flatten',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      axis=self.proto.flatten_param.axis
    )
    return [node], []

class Permute(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    perm = list(self.proto.permute_param.order)
    node = O.helper.make_node(
      op_type='Transpose',
      inputs=self.inputs,
      outputs=self.outputs,
      name=self.name,
      perm=perm
    )
    return [node], []
