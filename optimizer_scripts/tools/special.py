"""Special operations on model.
"""
import onnx.helper
import numpy as np
from . import helper
from . import other
from . import modhelper

def change_first_conv_from_bgr_to_rgb(m):
    """For input channel format BGR model, use this function to change the first
    conv weight to adapt the input into RGB.

    :param m: the model proto
    """
    # Check for first node.
    g = m.graph
    input_name = g.input[0].name
    first_nodes = helper.find_following_nodes_by_input_value_name(g, input_name)
    if len(first_nodes) > 1:
        return False
    first_node = first_nodes[0]
    # Now we have the first node. Check this first node.
    if first_node.op_type != 'Conv':
        return False
    weight_value = helper.find_value_by_name(g, first_node.input[1])
    weight_shape = helper.get_shape_from_value_info(weight_value)
    if weight_shape[1] != 3:
        return False
    # Do weight shuffle
    weight_node = helper.find_node_by_output_name(g, weight_value.name)
    weight_np = helper.constant_to_numpy(weight_node)
    b_channel = np.expand_dims(weight_np[:, 0, :, :], axis=1)
    g_channel = np.expand_dims(weight_np[:, 1, :, :], axis=1)
    r_channel = np.expand_dims(weight_np[:, 2, :, :], axis=1)
    new_np = np.concatenate((r_channel, g_channel, b_channel), axis=1)
    new_node = helper.numpy_to_constant(weight_value.name, new_np)
    # Replace the weight and topological sort
    g.node.remove(weight_node)
    g.node.extend([new_node])
    other.topological_sort(g)
    return True

def change_input_from_bgr_to_rgb(m):
    """For input channel format BGR model, use this function to modify the model
    to accepct RGB image.If the first node is a non-group Conv. Modify weight to
    adapt the input into RGB. Otherwise create a new node.

    :param m: the model proto
    """
    g = m.graph
    if len(g.input) > 1:
        print("This model has multiple inputs. Cannot change to RGB input.")
        return
    input_shape = helper.get_shape_from_value_info(g.input[0])
    if len(input_shape) != 4 or input_shape[1] != 3:
        print("The input shape is invalid for bgr conversion.")
        return
    # Try change conv weight first
    if change_first_conv_from_bgr_to_rgb(m):
        return
    # Otherwise, create a special conv node and replace the input
    # Construct weight
    weight_np = np.zeros((3, 3, 3, 3)).astype('float32')
    weight_np[0, 2, 1, 1] = 1.0
    weight_np[1, 1, 1, 1] = 1.0
    weight_np[2, 0, 1, 1] = 1.0
    new_weight = helper.numpy_to_constant("bgr_shuffle_weight", weight_np)
    # Construct Conv
    new_conv = onnx.helper.make_node(
        'Conv',
        ['rgb_input', "bgr_shuffle_weight"],
        [g.input[0].name],
        name='bgr_shuffle',
        dilations=[1, 1],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )
    # Connect the graph
    old_input_value = g.input.pop()
    new_input_value = onnx.helper.make_tensor_value_info(
        'rgb_input',
        old_input_value.type.tensor_type.elem_type,
        input_shape
    )
    g.input.extend([new_input_value])
    g.node.extend([new_weight, new_conv])
    # topological sort
    other.topological_sort(g)

def add_0_5_to_normalized_input(m):
    """For normalized input between -0.5 ~ 0.5, add 0.5 to the input to keep it
    between 0 ~ 1.

    :param m: the model proto
    """
    g = m.graph
    if len(g.input) > 1:
        print("This model has multiple inputs. Cannot normalize input.")
        return
    input_shape = helper.get_shape_from_value_info(g.input[0])
    if len(input_shape) != 4:
        print("The input shape is not BCHW. Cannot normalize input.")
        return
    # Construct weight
    ch = input_shape[1]
    weight_np = np.zeros((ch, ch, 3, 3)).astype('float32')
    for i in range(ch):
        weight_np[i, i, 1, 1] = 1.0
    new_weight = helper.numpy_to_constant("input_norm_weight", weight_np)
    # Construct bias
    bias_np = np.array([0.5] * ch).astype('float32')
    new_bias = helper.numpy_to_constant("input_norm_bias", bias_np)
    # Construct Conv
    new_conv = onnx.helper.make_node(
        'Conv',
        ['origin_input', "input_norm_weight", "input_norm_bias"],
        [g.input[0].name],
        name='input_norm',
        dilations=[1, 1],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )
    # Construct value_infos
    old_input_value = g.input.pop()
    weight_value = onnx.helper.make_tensor_value_info(
        'input_norm_weight',
        old_input_value.type.tensor_type.elem_type,
        [3, 3, 3, 3]
    )
    bias_value = onnx.helper.make_tensor_value_info(
        'input_norm_bias',
        old_input_value.type.tensor_type.elem_type,
        [3]
    )
    # Connect the graph
    new_input_value = onnx.helper.make_tensor_value_info(
        'origin_input',
        old_input_value.type.tensor_type.elem_type,
        input_shape
    )
    g.input.extend([new_input_value])
    g.node.extend([new_weight, new_bias, new_conv])
    g.value_info.extend([weight_value, bias_value, old_input_value])
    # topological sort
    other.topological_sort(g)

def add_rgb2yynn_node(m):
    """Add a conv layer which can convert rgb to yynn input.
    """
    g = m.graph
    if len(g.input) > 1:
        print("This model has multiple inputs. Cannot change to rgb input.")
        return
    input_shape = helper.get_shape_from_value_info(g.input[0])
    if len(input_shape) != 4:
        print("The input shape is not BCHW. Cannot normalize input.")
        return
    # Construct weight
    ch = input_shape[1]
    weight_np = np.zeros((3, 3, 4, 4)).astype('float32')
    weight_np[1, 1, :3, :2] = np.array([[[[0.299],
                                          [0.587],
                                          [0.114]]]])
    weight_np[1, 1, 3, 2:] = 1.
    weight_np = np.transpose(weight_np, (3, 2, 0, 1))
    new_weight = helper.numpy_to_constant("input_rgb2yynn_weight", weight_np)
    # Construct conv node
    new_conv = onnx.helper.make_node(
        'Conv',
        ['new_input', "input_rgb2yynn_weight"],
        [g.input[0].name],
        name='input_rgba2yynn',
        dilations=[1, 1],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )
    # Construct value_infos
    old_input_value = g.input.pop()
    weight_value = onnx.helper.make_tensor_value_info(
        'input_rgb2yynn_weight',
        old_input_value.type.tensor_type.elem_type,
        [4, 4, 3, 3]
    )
    # Connect the graph
    new_input_value = onnx.helper.make_tensor_value_info(
        'new_input',
        old_input_value.type.tensor_type.elem_type,
        input_shape
    )
    g.input.extend([new_input_value])
    g.node.extend([new_weight, new_conv])
    g.value_info.extend([weight_value, old_input_value])
    # topological sort
    other.topological_sort(g)
