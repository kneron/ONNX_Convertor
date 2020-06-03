
"""Converters for batch nomalization layers in Keras
"""
import onnx as O
import numpy as np
import logging

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

# Logger options and flags
logger = logging.getLogger("onnx-caffe")
logger.setLevel(logging.DEBUG)
batchReplace = False
noneDataFormat = False

class BatchNorm(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    bn_layer = self.layer[0]
    scale_layer = self.layer[1]

    node_list = []
    value_infos = []
    w_count = 0
    outname = self.outputs[0]
    # Weight 2 is tricky, mean and variance need to be divided by it
    moving_average_fractor = bn_layer.blobs[2].data[0]
    if moving_average_fractor == 0:
      moving_average_fractor = 1
    # Construct scale(gamma)
    gamma = scale_layer.blobs[0].data
    # print(gamma)
    #gamma = np.ones(scale_layer.blobs[0].data.shape, dtype = np.float32)
    if gamma.shape != bn_layer.blobs[0].data.shape:
      if len(gamma.shape) != 1:
        logger.warning("More than one dimension in Scale layer {}: {}".format(self.name, gamma.shape))
      else:
        if gamma.shape[0] % 2 != 0:
          logger.warning("The shape of gamma is not even")
        length = gamma.shape[0] // 2
        gamma = gamma[0: length]
    w_count += 1
    gamma_name = self.name + '_gamma'
    #print(gamma_name, gamma)
    tn, ti = helper.constructConstantNode(gamma_name, gamma)
    node_list += tn
    value_infos += ti
    self.inputs.append(gamma_name)
    # Construct bias(beta)
    # print(self.name)
    # print(len(scale_layer.blobs))
    beta = scale_layer.blobs[1].data
    #beta = np.zeros(scale_layer.blobs[1].data.shape, dtype = np.float32)
    if beta.shape != bn_layer.blobs[0].data.shape:
      if len(beta.shape) != 1:
        logger.warning("More than one dimension in Scale layer")
      else:
        if beta.shape[0] % 2 != 0:
          logger.warning("The shape of beta is not even")
        length = beta.shape[0] // 2
        beta = beta[0: length]
    w_count += 1
    beta_name = self.name + '_beta'
    #print(beta_name, beta)
    tn, ti = helper.constructConstantNode(beta_name, beta)
    node_list += tn
    value_infos += ti
    self.inputs.append(beta_name)
    # Construct mean
    mean = bn_layer.blobs[0].data
    mean = mean / moving_average_fractor
    w_count += 1
    mean_name = self.name + '_mean'
    #print(mean_name, mean)
    tn, ti = helper.constructConstantNode(mean_name, mean)
    node_list += tn
    value_infos += ti
    self.inputs.append(mean_name)
    # Construct var
    var = bn_layer.blobs[1].data
    var = var / moving_average_fractor
    var_name = self.name + '_var'
    #print(var_name, var)
    tn, ti = helper.constructConstantNode(var_name, var)
    node_list += tn
    value_infos += ti
    self.inputs.append(var_name)
    # Make the node, need to CHECK spatial later

    eps = self.proto.batch_norm_param.eps if self.proto.batch_norm_param.eps > 0 else 1e-5
    momentum = self.proto.batch_norm_param.moving_average_fraction if self.proto.batch_norm_param.moving_average_fraction > 0 else 0.999
    #print(self.name, eps, momentum)
    node = O.helper.make_node(
      op_type='BatchNormalization',
      inputs=self.inputs,
      outputs=[outname],
      name=self.name,
      #use the default epsilon and momentum
      epsilon = eps,
      momentum = momentum
      )
    node_list.append(node)
    # Construct Output
    '''
    if len(self.layer.input_shape) == 2:
      # Convert the no dimension case back
      info = O.helper.make_tensor_value_info(
        outname,
        helper.convertKerasType(helper.dtype),
        info_shape
      )
      value_infos.append(info)
      shape_name = self.name + '_flatten'
      self.inputs = [outname]
      node = O.helper.make_node(
        op_type='Flatten',
        inputs=self.inputs,
        outputs=[final_out],
        name=self.name + '_flatten'
      )
      node_list.append(node)
    '''
    return node_list, value_infos


class Power(Layer):
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    node_list = []
    node = O.helper.make_node(
      op_type       = 'Neg',
      inputs        = self.inputs,
      outputs       = self.outputs,
      name          = self.name,
      )

    node_list.append(node)
    return node_list, []

class Normalize(Layer):
  """Normalize layer is only in intel caffe.
  It is converted to a L2Norm and an element wise mul.
  """
  def __init__(self, inputs, outname, layer, proto, blob):
    Layer.__init__(self, inputs, outname, layer, proto, blob)
  def generate(self):
    #TODO: Support scale build from initializer
    node_list = []
    value_list = []
    # Construct weight
    scale = self.layer.blobs[0].data
    tn, ti = helper.constructConstantNode(self.name + '_scale', scale)
    node_list.extend(tn)
    value_list.extend(ti)
    # Construct L2Norm
    l2norm_node = O.helper.make_node(
      op_type     = 'LpNormalization',
      inputs      = self.inputs,
      outputs     = [self.name + '_norm_out'],
      name        = self.name,
      axis        = 1
    )
    l2norm_info = O.helper.make_tensor_value_info(
      self.name + '_norm_out',
      helper.convertKerasType(helper.dtype),
      self.blob.data.shape
    )
    node_list.append(l2norm_node)
    value_list.append(l2norm_info)
    # Construct scale
    scale_node = O.helper.make_node(
      op_type     = 'Mul',
      inputs      = [self.name + '_norm_out', self.name + '_scale'],
      outputs     = self.outputs,
      name        = self.name + '_mul',
    )
    node_list.append(scale_node)
    return node_list, value_list
