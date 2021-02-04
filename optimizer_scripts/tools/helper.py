"""This module contains helper functions that do not modify the graph.
"""
import onnx
import onnx.helper
import struct
import numpy as np

__ONNX_VERSION__ =  -1

def setup_current_opset_version(m):
    global __ONNX_VERSION__
    __ONNX_VERSION__ = m.opset_import[0].version
    if __ONNX_VERSION__ not in [9, 11]:
        raise RuntimeError('Only support opset 9 and 11, but got ' + str(__ONNX_VERSION__))

def get_current_opset_version():
    if __ONNX_VERSION__ == -1:
        raise RuntimeError('do setup_current_opset_version first please')
    return __ONNX_VERSION__

def find_nodes_by_input_name(g, name):
    nodes = []
    for node in g.node:
        if name in node.input:
            nodes.append(node)
    return nodes

def find_node_by_output_name(g, name):
    """
    Find a node in the graph by its output name

    :param g: the onnx graph\\
    :param name: the target node output name\\
    :returns: the node find by name
    """
    for i in g.node:
        if name in i.output:
            return i
    return None

def find_following_nodes_by_input_value_name(g, name):
    """ Find the following nodes of a specific value.

    :param g: the onnx graph. \\
    :param name: the value name. \\
    :return: a list of following nodes.
    """
    return find_nodes_by_input_name(g, name)

def find_value_by_name(g, name):
    """
    Find a value_info in the graph by name

    :param g: the onnx graph\\
    :param name: the target value_info name\\
    :returns: the value_info find by name
    """
    for i in g.value_info:
        if i.name == name:
            return i
    return None

def find_output_by_name(g, name):
    """
    Find a value_info in the graph by name

    :param g: the onnx graph\\
    :param name: the target value_info name\\
    :returns: the value_info find by name
    """
    for i in g.output:
        if i.name == name:
            return i
    return None

def find_input_by_name(g, name):
    """
    Find a input in the graph by name

    :param g: the onnx graph\\
    :param name: the target input name\\
    :returns: the input find by name
    """
    for i in g.input:
        if i.name == name:
            return i
    return None

def list_to_constant(name, shape, data, data_type=None):
    """Generate a constant node using the given infomation.

    :name: the node name and the output value name\\
    :shape: the data shape\\
    :data: the data itself\\
    :returns: the generated onnx constant node
    """
    if not data_type:
        if isinstance(data, int):
            data_type = onnx.helper.TensorProto.INT64
        elif isinstance(data, float):
            data_type = onnx.helper.TensorProto.FLOAT
        elif len(data) > 0 and isinstance(data[0], int):
            data_type = onnx.helper.TensorProto.INT64
        else:
            data_type = onnx.helper.TensorProto.FLOAT
    tensor = onnx.helper.make_tensor(
        name,
        data_type,
        shape,
        data
    )
    new_w_node = onnx.helper.make_node(
        "Constant",
        [],
        [name],
        name = name,
        value = tensor
    )
    return new_w_node

def numpy_to_constant(name, np_array):
    return list_to_constant(name, np_array.shape, np_array.flatten().tolist())

def constant_to_list(node):
    """Generate a list from the constant node

    :node: the Constant node\\
    :returns: the shape of the constant node, the data of the constant node
    """
    tensor = node.attribute[0].t
    # 1. check data type
    # 2. get data from raw or data
    # 3. get shape from dim
    if tensor.data_type == onnx.helper.TensorProto.INT32:
        if len(tensor.int32_data) != 0:
            data = list(tensor.int32_data)
        else:
            data = [i[0] for i in struct.iter_unpack('i', tensor.raw_data)]
    elif tensor.data_type == onnx.helper.TensorProto.INT64:
        if len(tensor.int64_data) != 0:
            data = list(tensor.int64_data)
        else:
            data = [i[0] for i in struct.iter_unpack('q', tensor.raw_data)]
    elif tensor.data_type == onnx.helper.TensorProto.FLOAT:
        if len(tensor.float_data) != 0:
            data = list(tensor.float_data)
        else:
            data = [i[0] for i in struct.iter_unpack('f', tensor.raw_data)]
    elif tensor.data_type == onnx.helper.TensorProto.DOUBLE:
        if len(tensor.double_data) != 0:
            data = list(tensor.double_data)
        else:
            data = [i[0] for i in struct.iter_unpack('d', tensor.raw_data)]
    else:
        print("Not supported data type {}".format(tensor.data_type))
        raise RuntimeError
    if len(tensor.dims) == 0:
        shape = len(data)
    else:
        shape = list(tensor.dims)
    return shape, data

def constant_to_numpy(node):
    """Generate a numpy array from the constant node

    :node: the Constant node\\
    :returns: the numpy array
    """
    shape, data = constant_to_list(node)
    return np.array(data).reshape(shape)

def all_constant_input(node):
    """Find the inputs of the given node. If the inputs of this node are all\\
    constant nodes, return True. Otherwise, return False.

    :param node: the input node which has a Node structure\\
    :return: whether the node of this node are all constant
    """
    if node.proto is None:
        return False
    isConstant = True
    for parent in node.parents:
        if parent.proto is None or parent.proto.op_type != 'Constant':
            isConstant = False
            break
    return isConstant

def get_padding(size, kernel_size, strides):
  """ Calculate the padding array for same padding in the Tensorflow fashion.\\
  See https://www.tensorflow.org/api_guides/python/nn#Convolution for more.
  """
  if size[0] % strides[0] == 0:
    pad_h = max(kernel_size[0] - strides[0], 0)
  else:
    pad_h = max(kernel_size[0] - (size[0] % strides[0]), 0)
  if size[1] % strides[1] == 0:
    pad_w = max(kernel_size[1] - strides[1], 0)
  else:
    pad_w = max(kernel_size[1] - (size[1] % strides[1]), 0)
  return [pad_h//2, pad_w//2, pad_h-pad_h//2, pad_w-pad_w//2]

def get_shape_from_value_info(value):
    """Get shape from a value info.

    :param value: the value_info proto\\
    :return: list of the shape
    """
    return [d.dim_value for d in value.type.tensor_type.shape.dim]

def find_size_shape_from_value(value):
    '''
    Find the size of data within the value_info object.
    :param value: value_info
    :return: int size and list shape of the data in the value_info
    '''
    if not value:
        return None, None
    if not value.type.tensor_type.shape.dim:
        return 0, []
    size = 1
    shape = []
    for i in range(len(value.type.tensor_type.shape.dim)):
        size *= max(1, value.type.tensor_type.shape.dim[i].dim_value)
        shape.append(max(1, value.type.tensor_type.shape.dim[i].dim_value))

    return size, shape


def get_attribute_by_name(node, attr_name):
    """Get attribute proto with specific name in the given node proto.

    :param node: the node proto.\\
    :param attr_name: a str for the name of the target.\\
    :return: if found, return the attribute_proto. Else, return None.
    """
    for attr in node.attribute:
        if attr.name == attr_name:
            return attr
    return None

def get_list_attribute_by_name(node, attr_name: str, attr_type: str):
    """Get list attribute with specific name in the given node proto.

    :param node: the node proto.\\
    :param attr_name: a str for the name of the target.\\
    :param attr_type: a str which should be "float" or "int".\\
    :return: if found, return the list. Else, return None.
    """
    attr_proto = get_attribute_by_name(node, attr_name)
    if attr_proto is None:
        return None
    if attr_type == "int":
        if len(attr_proto.ints) == 0:
            return None
        else:
            return list(attr_proto.ints)
    elif attr_type == "float":
        if len(attr_proto.ints) == 0:
            return None
        else:
            return list(attr_proto.floats)
    else:
        print("Warning: undefined type for list attribute extraction")
        return None

def get_var_attribute_by_name(node, attr_name: str, attr_type: str):
    """Get variable attribute with specific name in the given node proto.

    :param node: the node proto.\\
    :param attr_name: str for the name of the target.\\
    :param attr_type: str which should be "float", "int", "string" or "tensor".\\
    :return: if found, return the variable. Else, return None.
    """
    attr_proto = get_attribute_by_name(node, attr_name)
    if attr_proto is None:
        return None
    if attr_type == "int":
        return attr_proto.i
    elif attr_type == "float":
        return attr_proto.f
    elif attr_type == "string":
        if type(attr_proto.s) == type(b'abc'):
            return attr_proto.s.decode("utf-8")
        else:
            return attr_proto.s
    elif attr_type == "tensor":
        return attr_proto.t
    else:
        print("Warning: undefined type for variable attribute extraction")
        return None

def flatten_with_depth(data, depth):
    output = []
    if type(data) not in [type(np.array([1])), type([1])]:
        return [[data, 0]]
    for item in data:
        if type(item) not in [type(np.array([1])), type([1])]:
            output.append([item, depth+1])
        else:
            output += flatten_with_depth(item, depth+1)
    return output

def flatten_to_list(data):
    flatten_depth = flatten_with_depth(data, 0)
    flat_data = [item[0] for item in flatten_depth]
    return flat_data

def get_shape(data):
    shape = []
    if type(data) not in [type(np.array([1])), type([1])]:
        return []
    sub_data = data[0]
    shape.append(len(data))
    while type(sub_data) in [type(np.array([1])), type([1])]:
        shape.append(len(sub_data))
        sub_data = sub_data[0]
    return shape


def slice_data(data, starts, ends, axes):
    flat_data = [item[0] for item in flatten_with_depth(data, 0)]
    shape = get_shape(data)

    starts_updated = []
    ends_updated = []
    for i in range(len(starts)):
        start_updated = min(starts[i], shape[i]-1) % shape[i]
        starts_updated.append(start_updated)
    for j in range(len(starts)):
        if ends[j] >= shape[j]:
            end_updated = shape[j]
        else:
            end_updated = min(ends[j], shape[j]) % shape[j]
        ends_updated.append(end_updated)

    index_slices = []
    for i in range(len(shape)):
        if i not in axes:
            index_slices.append(list(range(shape[i])))
        else:
            axe_ind = axes.index(i)
            index_slices.append(list(range(starts_updated[axe_ind], ends_updated[axe_ind])))

    indices = [1]
    for i in range(len(shape)-1, -1, -1):
        step = np.prod(shape[i+1:])
        temp_pos = indices
        new_indices = []
        for n in index_slices[i]:
            for pos in temp_pos:
                new_indices.append(int(n*step+pos))
        indices = new_indices

    sliced_data = [flat_data[k-1] for k in indices]

    # reshape to correct shape.
    new_shape = []
    for i in range(len(shape)):
        if i not in axes:
            new_shape.append(shape[i])
        else:
            axe_ind = axes.index(i)
            new_shape.append(ends_updated[axe_ind]-starts_updated[axe_ind])
    if any([dim < 1 for dim in new_shape]):
        raise RuntimeError('Invalid starts ends.')
    
    sliced_data = np.reshape(sliced_data, new_shape)

    return sliced_data

def concatenate(data_sets, axis):
    # check shapes
    shapes = []
    shapes_ = []
    for data_set in data_sets:
      shape = get_shape(data_set)
      shapes.append(list(shape))
      shape.pop(axis)
      shapes_.append(shape)
    if not all([s == shapes_[0] for s in shapes_]):
      raise RuntimeError('data sets shapes do not match')
    
    new_dim = sum([s[axis] for s in shapes])
    new_shape = list(shapes[0])
    new_shape[axis] = new_dim

    flat_data_sets = []
    for data_set in data_sets:
      flat_data_sets.append(flatten_to_list(data_set))
    
    sub_block_size = 1
    for i in range(axis+1, len(shapes[0])):
      sub_block_size *= shapes[0][i]
    
    split_num = 1
    for i in range(axis):
      split_num *= shapes[0][i]

    total_flat_data = []
    for i in range(split_num):
      for j in range(len(shapes)):
        block_size = sub_block_size*shapes[j][axis]
        total_flat_data.extend(flat_data_sets[j][i*block_size:(i+1)*block_size])
    
    new_data = np.reshape(total_flat_data, new_shape)

    return new_data


def broadcast_data_sets(data_set_1, data_set_2):
    shape1 = get_shape(data_set_1)
    shape2 = get_shape(data_set_2)
    
    # compare shapes and get broadcasted shape
    list_a, list_b = (shape1, shape2) if len(shape1) > len(shape2) else (shape2, shape1)
    while len(list_a) > len(list_b):
        list_b.insert(0, 0)
    broadcasted_shape = []
    for i in range(len(list_a)):
      if list_b[i] == 0:
          broadcasted_shape.append(list_a[i])
      elif list_b[i] == 1:
          broadcasted_shape.append(list_a[i])
      elif list_a[i] == 1:
          broadcasted_shape.append(list_b[i])
      elif list_a[i] == list_b[i]:
          broadcasted_shape.append(list_a[i])
      else:
          raise RuntimeError('Can not broadcast two data sets')

    # prepare data for broadcasting.
    shape1 = list(map(lambda x:x if x != 0 else 1, shape1))
    shape2 = list(map(lambda x:x if x != 0 else 1, shape2))
    data_1 = np.reshape(data_set_1, shape1)
    data_2 = np.reshape(data_set_2, shape2)

    for i in range(len(shape1)):
      if shape1[i] != broadcasted_shape[i]:
        new_data_total = [list(data_1) for _ in range(broadcasted_shape[i])]
        data_1 = concatenate(new_data_total, axis=i)
    for i in range(len(shape2)):
      if shape2[i] != broadcasted_shape[i]:
        new_data_total = [list(data_2) for _ in range(broadcasted_shape[i])]
        data_2 = concatenate(new_data_total, axis=i)

    return data_1, data_2


def add(data_set_1, data_set_2):
  broadcasted_data_1, broadcasted_data_2 = broadcast_data_sets(data_set_1, data_set_2)

  flat_data_1 = flatten_to_list(broadcasted_data_1)
  flat_data_2 = flatten_to_list(broadcasted_data_2)
  shape = get_shape(broadcasted_data_1)
  res = []
  for i in range(len(flat_data_1)):
    res.append(flat_data_1[i]+flat_data_2[i])
  
  res = np.reshape(res, shape)

  return res


def reduceprod(data_set, axis, keepdims=1):
  flat_data = flatten_to_list(data_set)
  old_shape = get_shape(data_set)

  temp_shape = old_shape
  temp_flat_data = flat_data
  for ax in axis:
    split_num = 1
    step = 1
    for i in range(ax):
      split_num *= temp_shape[i]
    for i in range(ax+1, len(temp_shape)):
      step *= temp_shape[i]
    
    block_size = len(temp_flat_data)//split_num
    new_flat_data = []
    for j in range(split_num):
      block_data = temp_flat_data[j*block_size:(j+1)*block_size]
      reduced_block_data = []
      for k in range(step):
        val = block_data[k]
        for l in range(1, block_size//step):
          val *= block_data[k+l*step]
        reduced_block_data.append(val)
      new_flat_data.extend(reduced_block_data)
    temp_flat_data = new_flat_data
    temp_shape[ax] = 1
  
  new_flat_data = temp_flat_data
  new_shape = temp_shape
  if not keepdims:
    axis = sorted(list(axis))
    for pos in axis[::-1]:
      new_shape.pop(pos)
  
  return np.reshape(new_flat_data, new_shape)


def transpose(data_set, permutation):
  # find series of local swaps
  data_set = list(data_set)
  perm = list(permutation)
  shape = get_shape(data_set)
  flat_data = flatten_to_list(data_set)
  assert set(perm) == set(range(len(shape))), 'invalid permutation'

  new_shape = [shape[i] for i in perm]
  swaps = []
  bubbled = True
  while bubbled:
    bubbled = False
    for i in range(len(new_shape)-1):
      if perm[i] > perm[i+1]:
        swaps.append([i, i+1])
        p_1, p_2 = perm[i], perm[i+1]
        perm[i], perm[i+1] = p_2, p_1
        bubbled = True
  
  # apply local swaps
  current_shape = list(shape)
  temp_flat_data = flat_data

  for swap in swaps[::-1]:
    ind_1, ind_2 = swap[0], swap[1]
    dim_1 = current_shape[ind_1]
    dim_2 = current_shape[ind_2]
    split_num = 1
    block_size = 1

    for i in range(ind_1):
      split_num *= current_shape[i]
    for i in range(ind_2+1, len(current_shape)):
      block_size *= current_shape[i]

    data_blocks = np.reshape(temp_flat_data, [-1, block_size])
    flat_data_1 = []
    for k in range(split_num):
      block = []
      for m in range(dim_2):
        for n in range(dim_1):
          block_pos = k*dim_1*dim_2 + n*dim_2+m
          block.extend(data_blocks[block_pos])
      flat_data_1.extend(block)

    temp_flat_data = flat_data_1
    current_shape[ind_1] = dim_2
    current_shape[ind_2] = dim_1

  return np.reshape(temp_flat_data, current_shape)

def subtract(data_set_1, data_set_2):
    broadcasted_data_1, broadcasted_data_2 = broadcast_data_sets(data_set_1, data_set_2)

    shape = get_shape(broadcasted_data_1)
    flat_data_1 = flatten_to_list(broadcasted_data_1)
    flat_data_2 = flatten_to_list(broadcasted_data_2)

    substracted_data = [flat_data_1[i] - flat_data_2[i] for i in range(len(flat_data_1))]

    new_data = np.reshape(substracted_data, shape)

    return new_data

    