
"""Converters for batch nomalization layers in Keras
"""
import onnx as O
import numpy as np

from .base_layer import Layer
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import helper

class BatchNormalization(Layer):
  def __init__(self, node):
    Layer.__init__(self, node)
  def generate(self):
    node_list = []
    value_infos = []
    w_count = 0
    outname = self.outputs[0]
    # Construct scale(gamma)
    if self.layer.gamma is None:
      gamma = np.ones(
        self.layer.moving_mean.shape,
        dtype=self.layer.get_weights()[0].dtype
      )
      gamma_name = self.name + '_gamma'
      tn, ti = helper.constructConstantNode(gamma_name, gamma)
    else:
      if helper.duplicate_weights:
        gamma_name = self.name + '_gamma'
      else:
        gamma_name = self.layer.weights[w_count].name
      gamma = self.layer.get_weights()[w_count]
      tn, ti = helper.getConstantNodeByName(gamma_name, gamma)
      w_count += 1
    node_list += tn
    value_infos += ti
    self.inputs.append(gamma_name)
    # Construct bias(beta)
    if self.layer.beta is None:
      beta = np.zeros(
        self.layer.moving_mean.shape,
        dtype=self.layer.get_weights()[0].dtype
      )
      beta_name = self.name + '_beta'
      tn, ti = helper.constructConstantNode(beta_name, beta)
    else:
      if helper.duplicate_weights:
        beta_name = self.name + "_beta"
      else:
        beta_name = self.layer.weights[w_count].name
      beta = self.layer.get_weights()[w_count]
      tn, ti = helper.getConstantNodeByName(beta_name, beta)
      w_count += 1
    node_list += tn
    value_infos += ti
    self.inputs.append(beta_name)
    # Construct mean
    if helper.duplicate_weights:
      mean_name = self.name + "_mean"
    else:
      mean_name = self.layer.weights[w_count].name
    mean = self.layer.get_weights()[w_count]
    w_count += 1
    tn, ti = helper.getConstantNodeByName(mean_name, mean)
    node_list += tn
    value_infos += ti
    self.inputs.append(mean_name)
    # Construct var
    if helper.duplicate_weights:
      var_name = self.name + "_var"
    else:
      var_name = self.layer.weights[w_count].name
    var = self.layer.get_weights()[w_count]
    tn, ti = helper.getConstantNodeByName(var_name, var)
    node_list += tn
    value_infos += ti
    self.inputs.append(var_name)
    # Make the node, need to CHECK spatial later
    node = O.helper.make_node(
      op_type='BatchNormalization',
      inputs=self.inputs,
      outputs=[outname],
      name=self.name,
      epsilon=float(self.layer.epsilon),
      momentum=float(self.layer.momentum)
      )
    node_list.append(node)
    return node_list, value_infos