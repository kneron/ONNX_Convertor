from . import helper
from . import other
from . import modhelper
from . import fusing
import numpy as np
import onnx
import onnx.utils

def eliminate_transposes(m):
  g = m.graph
  keep_eliminating = True
  while keep_eliminating:
    while swap_transpose_with_single_next_node(g):
      pass
    splitted = split_transpose_for_multiple_next_nodes(g)
    annihilated = annihilate_transposes(g)
    multiple_trans_swapped = swap_multiple_transposes_with_node(g)
    keep_eliminating = splitted or annihilated or multiple_trans_swapped

    if keep_eliminating:
      m = onnx.utils.polish_model(m)
      g = m.graph
  
  return m


def swap_transpose_with_single_next_node(g):
  swapped = False
  passable_nodes = set(['Relu', 'Neg', 'LeakyRelu', 'Sqrt', 'Reciprocal', 'Add', 'Mul', 'Tanh'])
  for node in g.node:
    trans_node = node
    # Check for transpose node
    if trans_node.op_type != 'Transpose':
      continue
    next_nodes = helper.find_nodes_by_input_name(g, trans_node.output[0])
    if len(next_nodes) != 1:
      continue
    next_node = next_nodes[0]
    # Check if the next node is the type can be swapped
    if next_node.op_type not in passable_nodes:
      continue

    input_nodes = [helper.find_node_by_output_name(g, input_name) for input_name in next_node.input]

    # Check if the node has nonconstant input other than the Transpose node itself
    nonconstant_input = False
    for input_node in input_nodes:
      if input_node == None:
        nonconstant_input = True
        break
      if input_node.name == trans_node.name:
        continue
      elif input_node.op_type == 'Constant':
        continue
      else:
        nonconstant_input = True
        break
    if nonconstant_input:
      continue

    for input_node in input_nodes:
      if input_node.name == trans_node.name:
        # if the input is just the transpose node
        next_value_info = helper.find_value_by_name(g, next_node.output[0])
        mid_value_info = helper.find_value_by_name(g, trans_node.output[0])

        output_nodes = helper.find_nodes_by_input_name(g, next_node.output[0])
        for out_node in output_nodes:
          modhelper.replace_node_input(out_node, next_node.output[0], trans_node.name)

        next_node.input[0] = trans_node.input[0]
        next_node.output[0] = next_node.name
        trans_node.input[0] = next_node.name
        trans_node.output[0] = trans_node.name

        if next_value_info:
          next_value_info.name = trans_node.name
        if mid_value_info:
          g.value_info.remove(mid_value_info)
      else:
        # if the input is a constant node
        old_tensor = input_node.attribute[0].t
        old_shape, data = helper.constant_to_list(input_node)
        # If the constant node is a scaler, no action is needed
        if type(old_shape) == int:
            old_shape = [old_shape]
        permutation = list(trans_node.attribute[0].ints)
        while len(old_shape) < len(permutation):
          old_shape.insert(0, 1)
        np_data = np.reshape(data, old_shape)
        reverse_perm = []
        for i in range(len(permutation)):
          reverse_perm.append(permutation.index(i))
        np_data = np.transpose(np_data, reverse_perm)
        new_shape = np_data.shape
        new_tensor = onnx.helper.make_tensor(
          name=old_tensor.name,
          data_type=old_tensor.data_type,
          dims=new_shape,
          vals=np_data.flatten().tolist()
        )
        new_node = onnx.helper.make_node(
          'Constant',
          [],
          [input_node.output[0]],
          name=input_node.name,
          value=new_tensor
        )
        g.node.extend([new_node])

        g.value_info.remove(helper.find_value_by_name(g, input_node.output[0]))
        g.node.remove(input_node)

    swapped = True

  other.topological_sort(g)
  return swapped


def swap_multiple_transposes_with_node(g):
  # here only consider same input transposes
  swapped = False
  passable_nodes = set(['Add', 'Mul'])
  node_to_del = []
  for node in g.node: 
    if node.op_type not in passable_nodes:
      continue
    input_nodes = [helper.find_node_by_output_name(g, input_name) for input_name in node.input]
    if any([input_node == None for input_node in input_nodes]):
      continue
    if any([input_node.op_type != 'Transpose' for input_node in input_nodes]):
      continue

    permutation = list(input_nodes[0].attribute[0].ints)
    if any([list(input_node.attribute[0].ints) != permutation for input_node in input_nodes]):
      continue
    
    for input_name in node.input:
      input_node = helper.find_node_by_output_name(g, input_name)
      modhelper.replace_node_input(node, input_name, input_node.input[0]) 

    node_to_del.extend(input_nodes)
    for input_node in input_nodes:
      input_val_info = helper.find_value_by_name(g, input_node.output[0])
      if input_val_info is not None:
        g.value_info.remove(input_val_info)
    output_val_info = helper.find_value_by_name(g, node.output[0])
    if output_val_info is not None:
      g.value_info.remove(output_val_info)

    output_nodes = helper.find_nodes_by_input_name(g, node.output[0])
    for i in range(len(output_nodes)):
      new_trans_node_name = node.name+'_trans_'+str(i)
      new_trans_node = onnx.helper.make_node(
        'Transpose',
        [node.output[0]],
        [new_trans_node_name],
        name=new_trans_node_name,
        perm=permutation
      )
      modhelper.replace_node_input(output_nodes[i], node.output[0], new_trans_node_name)
      
      g.node.extend([new_trans_node])
    
    swapped = True    
  
  while node_to_del:
    node = node_to_del.pop()
    g.node.remove(node)
  
  other.topological_sort(g)
  return swapped


def annihilate_transposes(g):
  node_to_del = []
  annihilated = False
  for node in g.node:
    if node.op_type != 'Transpose':
      continue
    pre_node = helper.find_node_by_output_name(g, node.input[0])
    if not pre_node or pre_node.op_type != 'Transpose':
      continue
    nodes_from_top_transpose = helper.find_nodes_by_input_name(g, pre_node.output[0])
    if len(nodes_from_top_transpose) > 1:
      continue
  
    perm_1 = list(pre_node.attribute[0].ints)
    perm_2 = list(node.attribute[0].ints)
    if perm_1 != perm_2:
      continue

    out_nodes = helper.find_nodes_by_input_name(g, node.output[0])
    for out_node in out_nodes:
      modhelper.replace_node_input(out_node, node.output[0], pre_node.input[0])
    
    node_to_del.extend([node, pre_node])
    mid_value_info = helper.find_value_by_name(g, pre_node.output[0])
    out_value_info = helper.find_value_by_name(g, node.output[0])
    g.value_info.remove(mid_value_info)
    g.value_info.remove(out_value_info)

    annihilated = True
  while node_to_del:
    node = node_to_del.pop()
    g.node.remove(node)
  
  return annihilated


def split_transpose_for_multiple_next_nodes(g):
  splitted = False
  node_to_del = []
  for node in g.node:
    if node.op_type != 'Transpose':
      continue
    output_nodes = helper.find_nodes_by_input_name(g, node.output[0])
    if len(output_nodes) < 2:
      continue
    for i in range(len(output_nodes)):
      output_node = output_nodes[i]
      new_trans_node_name = node.name + '_' + str(i)
      new_trans_node = onnx.helper.make_node(
        'Transpose',
        [node.input[0]],
        [new_trans_node_name],
        name=new_trans_node_name,
        perm=list(node.attribute[0].ints)
      )
      modhelper.replace_node_input(output_node, node.output[0], new_trans_node.output[0])
      g.node.extend([new_trans_node])
    
    node_to_del.append(node)
    val_info = helper.find_value_by_name(g, node.output[0])
    g.value_info.remove(val_info)

    splitted = True
  
  while node_to_del:
    node = node_to_del.pop()
    g.node.remove(node)
  
  other.topological_sort(g)
  return splitted

def remove_trivial_transpose(g):
  node_to_del = []
  for node in g.node:
    if node.op_type != 'Transpose':
      continue
    permutation = list(node.attribute[0].ints)
    if permutation != list(range(len(permutation))):
      continue
     
    next_nodes = helper.find_nodes_by_input_name(g, node.output[0])
    if not next_nodes:
      input_val_info = helper.find_value_by_name(g, node.input[0])
      out_val_info = helper.find_output_by_name(g, node.output[0])
      if not input_val_info:
        input_val_info = helper.find_input_by_name(g, node.input[0])
      g.output.remove(out_val_info)
      g.output.extend([input_val_info])
    else:
      out_val_info = helper.find_value_by_name(g, node.output[0])
      for next_node in next_nodes:
        modhelper.replace_node_input(next_node, node.output[0], node.input[0])
      g.value_info.remove(out_val_info)
    
    node_to_del.append(node)
  
  while node_to_del:
    node = node_to_del.pop()
    g.node.remove(node)
  
  other.topological_sort(g)

def fuse_Transpose_into_Gemm_weight(g):
  node_to_del = []
  for node in g.node:
    # Check pattern
    if node.op_type != 'Gemm':
      continue
    prev_node = helper.find_node_by_output_name(g, node.input[0])
    if prev_node is None or prev_node.op_type != 'Flatten':
      continue
    transpose_node = helper.find_node_by_output_name(g, prev_node.input[0])
    if transpose_node.op_type != 'Transpose':
      continue
    # Check attribute
    perm = helper.get_list_attribute_by_name(transpose_node, 'perm', 'int')
    if perm != [0, 2, 3, 1]:
      continue
    transB = helper.get_var_attribute_by_name(node, 'transB', 'int')
    if transB is not None and transB == 1:
      continue
    # Get the original weight
    origin_weight = helper.find_node_by_output_name(g, node.input[1])
    origin_np = helper.constant_to_numpy(origin_weight)
    # Calculate a new weight
    shape = helper.get_shape_from_value_info(helper.find_value_by_name(g, prev_node.input[0]))
    shape.append(-1)
    new_np = np.reshape(origin_np, shape)
    new_np = np.transpose(new_np, [0, 3, 1, 2, 4])
    new_np = np.reshape(new_np, [-1, new_np.shape[-1]])
    new_weight = helper.numpy_to_constant(origin_weight.output[0], new_np)
    # Replace and eliminate
    prev_node.input[0] = transpose_node.input[0]
    node_to_del.append(transpose_node)
    node_to_del.append(origin_weight)
    g.value_info.remove(helper.find_value_by_name(g, transpose_node.output[0]))
    g.node.extend([new_weight])

  while node_to_del:
    node = node_to_del.pop()
    g.node.remove(node)

  other.topological_sort(g)
