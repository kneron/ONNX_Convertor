from . import helper
from .exceptions import FeatureNotImplemented
import numpy as np

def fuse_bn_into_conv(layer_tree):
  remove_list = []
  for i in range(len(layer_tree)):
    node = layer_tree[i]
    # Find BN layers can be fused
    if node.type != "BatchNormalization":
      continue
    if node.inputs[0].input.type != "Conv2D":
      continue
    bn_layer = node
    conv_layer = node.inputs[0].input
    helper.logger.debug("Fuse " + bn_layer.name +" into " + conv_layer.name)

    # Prepare old weights
    # convolution weight and bias
    conv_w = np.transpose(conv_layer.klayer.get_weights()[0], [3,2,0,1])
    if conv_layer.klayer.use_bias:
      conv_b = conv_layer.klayer.get_weights()[1]
    else:
      conv_b = np.zeros(
        bn_layer.klayer.moving_mean.shape,
        conv_layer.klayer.get_weights()[0].dtype
      )
    # bn scale
    bn_weight_cnt = 0
    if bn_layer.klayer.gamma is None:
      bn_s = np.ones(
        bn_layer.klayer.moving_mean.shape,
        dtype=bn_layer.klayer.get_weights()[0].dtype
      )
    else:
      bn_s = bn_layer.klayer.get_weights()[bn_weight_cnt]
      bn_weight_cnt += 1
    # bn bias
    if bn_layer.klayer.beta is None:
      bn_b = np.zeros(
        bn_layer.klayer.moving_mean.shape,
        dtype=bn_layer.klayer.get_weights()[0].dtype
      )
    else:
      bn_b = bn_layer.klayer.get_weights()[bn_weight_cnt]
      bn_weight_cnt += 1
    # bn mean
    bn_m = bn_layer.klayer.get_weights()[bn_weight_cnt]
    bn_weight_cnt += 1
    # bn var
    bn_v = bn_layer.klayer.get_weights()[bn_weight_cnt]
    # bn epsilon
    bn_e = bn_layer.klayer.epsilon

    # Calculate new weights
    conv_w = conv_w.transpose([1, 2, 3, 0])
    new_w = np.multiply(conv_w, bn_s / (np.sqrt(bn_v + bn_e)))
    new_w = new_w.transpose([3, 0, 1, 2])
    new_b = (conv_b - bn_m) * bn_s / (np.sqrt(bn_v + bn_e)) + bn_b
    # Save the weights
    conv_layer.new_w = new_w
    conv_layer.new_b = new_b
    # Reconnect graph by set the output of BN as the output of conv
    conv_layer.outputs = bn_layer.outputs
    conv_layer.outputs[0].input = conv_layer
    remove_list.insert(0, i)

  # Remove origin bn node
  for index in remove_list:
    del layer_tree[index]

def eliminate_dropout(layer_tree):
  remove_list = []
  for i in range(len(layer_tree)):
    node = layer_tree[i]
    # Find if it is the Dropout layer
    if node.type != "Dropout":
      continue
    helper.logger.debug("Eliminating Dropout layer " + node.name)
    remove_list.insert(0, i)
    # Reconnect graph by reset the inputs of the nodes followed by Dropout
    input_tensor = node.inputs[0]
    input_tensor.outputs = []
    for following_node in node.outputs[0].outputs:
      following_node.replace_input(input_tensor, node.outputs[0])
      input_tensor.outputs.append(following_node)

  # Remove dropout layers
  for index in remove_list:
    del layer_tree[index]

def replace_average_pool(layer_tree):
  for node in layer_tree:
    # Check for the nodes need to replace
    if node.type != "AveragePooling2D":
      continue
    if node.klayer.padding != 'valid':
      continue
    if helper.data_format is None or helper.data_format == 'channels_last':
      if node.klayer.input_shape[1:3] != node.klayer.pool_size or node.klayer.output_shape[1:3] != (1, 1):
        continue
    else:
      if node.klayer.input_shape[2:4] != node.klayer.pool_size or node.klayer.output_shape[1:3] != (1, 1):
        continue
    # Check whether the next layer is Flatten
    skip = False
    if len(node.outputs[0].outputs) != 0:
      for following_node in node.outputs[0].outputs:
        if following_node.type != "Flatten":
          skip = True
          break
    if skip:
      continue
    # This node should be replaced
    helper.logger.debug("Replace AveragePooling layer " + node.name)
    node.type = "GlobalAveragePooling2D"
    # Reconnect the graph
    if len(node.outputs[0].outputs) == 0:
      # Reshape the final output
      helper.final_output_change.append(node.outputs[0].name)
    else:
      # Check whether the next layer is Flatten
      for following_node in node.outputs[0].outputs:
        if following_node.type != "Flatten":
          raise FeatureNotImplemented("AveragePooling with 1x1 output but without following Flatten layer")

def fuse_pad_into_next(layer_tree):
  remove_list = []
  merged_type = ['Conv2D', 'MaxPooling2D', 'AveragePooling2D']
  for i in range(len(layer_tree)):
    node = layer_tree[i]
    # Check for padding layers need to be fused
    if node.type != "ZeroPadding2D":
      continue
    if len(node.outputs[0].outputs) != 1 or node.outputs[0].outputs[0].type not in merged_type:
      continue
    # Find a node to fuse
    fused_node = node.outputs[0].outputs[0]
    helper.logger.debug("Fusing {}({}) into {}({})".format(node.name, node.type, fused_node.name, fused_node.type))
    # Reconnect the input
    the_input = node.inputs[0]
    the_input.replace_output([fused_node], node)
    fused_node.replace_input(the_input, node.outputs[0])
    # Setting up the extra padding
    fused_node.extra_attr = node.klayer.padding
    remove_list.insert(0, i)
  # Remove pad nodes
  for index in remove_list:
    del layer_tree[index]

pass_list = []
pass_list.append([eliminate_dropout])
pass_list.append([fuse_pad_into_next, replace_average_pool])
pass_list.append([fuse_bn_into_conv])
