"""Preprocessing mainly cares about constructing a DAG before we start processing.
The shared layers and models are also duplicated duing this procedue.
"""
import logging
import onnx as O
import keras as K
from keras.models import Sequential

from . import helper
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from .tree_structure import TreeNode, TreeTensor
from . import optimizer

def is_same_tensor(tensor_a, tensor_b):
  """Compare tow tree tensor class and check if their inner tensor is the same
  """
  return tensor_a.tensor == tensor_b.tensor

# Define the outside input and output as two special tree nodes
GRAND_INPUT = TreeTensor('INPUT')

def createTreeNode(layer_tree, tensor_dict, klayer, prefix=''):
  """The function used to create the tree node and tree tensors corresponding
     to the given keras layer.
  """
  total_use_num = len(klayer._inbound_nodes)
  if total_use_num == 1:
    # For non-shared layers
    cur_node = TreeNode(klayer, prefix)
    for i in range(len(cur_node.get_output_tensors())):
      ktensor = cur_node.get_output_tensors()[i]
      cur_tensor = TreeTensor(cur_node.name + '_o' + str(i),
          tensor=ktensor,
          creator=cur_node)
      tensor_dict[ktensor] = cur_tensor
      cur_node.outputs.append(cur_tensor)
    layer_tree.append(cur_node)
  else:
    # For shared layers
    for i in range(total_use_num):
      postfix = '_p' + str(i)
      cur_node = TreeNode(klayer, prefix, postfix, i)
      for j in range(len(cur_node.get_output_tensors())):
        ktensor = cur_node.get_output_tensors()[j]
        cur_tensor = TreeTensor(cur_node.name + '_o' + str(j),
            tensor=ktensor,
            creator=cur_node)
        tensor_dict[ktensor] = cur_tensor
        cur_node.outputs.append(cur_tensor)
      layer_tree.append(cur_node)

def check_rnn_start_point(layers):
  RNN_LAYERS = ['GRU', 'LSTM']
  START_LAYERS = ['Reshape', 'InputLayer']
  # Check for RNN layers
  the_first_rnn_layer = None
  for layer in layers:
    if layer.type in RNN_LAYERS:
      the_first_rnn_layer = layer
      break
  if the_first_rnn_layer is None:
    return
  # Find the start layer before this RNN layer
  layer = the_first_rnn_layer
  while layer.type not in START_LAYERS:
    layer = layer.inputs[0].input
  helper.RNN_start_node = layer
  if layer.type == 'InputLayer':
    helper.RNN_start = True
  helper.logger.debug("Found RNN starting from " + layer.name)

def preprocess(kmodel, prefix='', outer_node=None, optimize=False):
  """The main tree construction function
  """
  output_tensors = kmodel.outputs
  input_tensors = kmodel.inputs
  layer_tree = []
  input_node_list = []
  output_tensor_list = []
  submodel_list = []
  useless_node_idx_list = []
  helper.is_sequential = isinstance(kmodel, Sequential)

  # Define a dictionary to map the output tensors to a single tree tenser
  tensor_dict = dict()
  # Dictionary for input tensor
  input_dict = dict()

  # 1. Set up tree nodes and their tensors for all the layers
  for layer in kmodel.layers:
    createTreeNode(layer_tree, tensor_dict, layer, prefix=prefix)

  # 2.1 If it is sequential without InputLayer, construct one for it
  if helper.is_sequential:
    helper.logger.warning("Sequential model is not recommanded.")
    if layer_tree[0].type != "InputLayer":
      helper.logger.warning("Constructing an InputLayer for Sequential model.")
      constructed_input = TreeNode()
      constructed_input.name = "contructed_input"
      constructed_input.type = "InputLayer"
      actual_input_tensor = input_tensors[0]
      input_tensor = TreeTensor(constructed_input.name + '_o0',
          tensor=actual_input_tensor,
          creator=constructed_input)
      tensor_dict[actual_input_tensor] = input_tensor
      constructed_input.outputs.append(input_tensor)
    layer_tree.insert(0, constructed_input)

  # 2.2 Set up tree nodes inputs and outputs
  for tree_node in layer_tree:
    if tree_node.type == "InputLayer":
      # Input node has no regular input tensor
      if outer_node is None:
        tree_node.inputs.append(GRAND_INPUT)
        GRAND_INPUT.outputs.append(tree_node)
      else:
        tree_node.type = "InnerInput"
      keras_tensor = tree_node.get_input_tensors()[0]
      input_dict[keras_tensor] = tree_node
    else:
      # None input nodes
      try:
        for tensor in tree_node.get_input_tensors():
          src_tensor = tensor_dict[tensor]
          src_tensor.outputs.append(tree_node)
          tree_node.inputs.append(src_tensor)
      except KeyError:
        useless_node_idx_list.append(layer_tree.index(tree_node))
    # Check for submodel
    if tree_node.type == "Model":
      submodel_list.append(tree_node)

  # 3.1 Check for output nodes
  for output_tensor in output_tensors:
    output_tensor_list.append(tensor_dict[output_tensor])
  # 3.2 Check for input nodes
  try:
    for input_tensor in input_tensors:
      input_node_list.append(input_dict[input_tensor])
  except KeyError:
    raise FeatureNotImplemented("Keras model without Input layer")

  # 4. Deal with submodels
  for model in submodel_list:
    if layer_tree.index(model) in useless_node_idx_list:
      continue
    # 4.1 Extract submodel
    sub_layers, sub_inputs, sub_outputs = preprocess(model.klayer,
        prefix=model.name + '_',
        outer_node=model)
    layer_tree += sub_layers
    # 4.2 Reset inner input node
    assert len(model.inputs) == len(sub_inputs), "Submodel input number error"
    for i in range(len(sub_inputs)):
      # for each inner input layer
      input_out_tensor = sub_inputs[i].outputs[0]
      for input_follower in input_out_tensor.outputs:
        # For each use of the current input
        input_follower.replace_input(model.inputs[i], input_out_tensor)
      model.inputs[i].replace_output(input_out_tensor.outputs, model)
      useless_node_idx_list.append(layer_tree.index(sub_inputs[i]))
    # 4.3 Reset inner output node
    assert len(model.outputs) == len(sub_outputs), "Submodel output number error"
    for i in range(len(sub_outputs)):
      # for each output tensor
      sub_outputs[i].outputs += model.outputs[i].outputs
      for follower in model.outputs[i].outputs:
        # for each use of output tensor
        follower.replace_input(sub_outputs[i], model.outputs[i])
      if model.outputs[i] in output_tensor_list:
        sub_outputs[i].tensor = model.outputs[i].tensor
        output_tensor_list.remove(model.outputs[i])
        output_tensor_list.append(sub_outputs[i])
    useless_node_idx_list.append(layer_tree.index(model))

  # 5. Remove useless nodes
  for i in sorted(useless_node_idx_list, reverse=True):
    helper.logger.debug("Remove %s", layer_tree[i].name)
    del layer_tree[i]

  # 6. Optimizations
  for i in range(optimize):
    for opt_func in optimizer.pass_list[i]:
      opt_func(layer_tree)

  # Check RNN
  check_rnn_start_point(layer_tree)

  # 7. Only the out most preprocess will print the debug message
  if prefix != '':
    return layer_tree, input_node_list, output_tensor_list
  helper.logger.debug("Here goes the graph:")
  for layer in layer_tree:
    layer.print_info()
  for tensor in output_tensor_list:
    helper.logger.debug(tensor.name + '\t-> OUTPUT')

  return layer_tree, input_node_list, output_tensor_list