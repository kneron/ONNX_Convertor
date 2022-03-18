"""Special operations on model.
"""
import logging
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

def swap_MatMul_inputs(g, original_matmul_node):
    # Create Transpose nodes
    input_a_value = helper.find_value_by_name(g, original_matmul_node.input[0])
    input_a_shape = helper.get_shape_from_value_info(input_a_value)
    if len(input_a_shape) == 2:
        perm = [1, 0]
    else:
        perm = [0, 2, 1]
    new_input_b_node = onnx.helper.make_node(
        'Transpose',
        inputs = [input_a_value.name],
        outputs = [input_a_value.name + '_transposed'],
        name = f"{input_a_value.name}_transposed_for_{original_matmul_node.name}",
        perm = perm
    )
    input_b_value = helper.find_value_by_name(g, original_matmul_node.input[1])
    input_b_shape = helper.get_shape_from_value_info(input_b_value)
    if len(input_b_shape) == 3:
        perm = [0, 2, 1]
    else:
        perm = [0, 1, 3, 2]
    new_input_a_node = onnx.helper.make_node(
        'Transpose',
        inputs = [input_b_value.name],
        outputs = [input_b_value.name + '_transposed'],
        name = f'{input_b_value.name}_transposed_for_{original_matmul_node.name}',
        perm = perm
    )
    # Create new MatMul node
    new_matmul_node = onnx.helper.make_node(
        'MatMul',
        inputs = [new_input_a_node.output[0], new_input_b_node.output[0]],
        outputs = [original_matmul_node.output[0] + '_transposed'],
        name = original_matmul_node.name + '_transposed'
    )
    # Create final Transpose node
    output_value = helper.find_value_by_name(g, original_matmul_node.output[0])
    output_shape = helper.get_shape_from_value_info(output_value)
    if len(output_shape) == 3:
        perm = [0, 2, 1]
    else:
        perm = [0, 1, 3, 2]
    new_final_transpose_node = onnx.helper.make_node(
        'Transpose',
        inputs = [new_matmul_node.output[0]],
        outputs = [original_matmul_node.output[0]],
        name = original_matmul_node.name + '_final_transpose',
        perm = perm
    )
    # Add new nodes
    g.node.extend([new_input_a_node, new_input_b_node, new_matmul_node, new_final_transpose_node])
    # Delete original nodes
    g.node.remove(original_matmul_node)

def split_MatMul_batch_then_concat(g, original_matmul_node):
    new_nodes = []
    final_concat_inputs = []
    # Get the batch count
    input_a_value = helper.find_value_by_name(g, original_matmul_node.input[0])
    input_a_shape = helper.get_shape_from_value_info(input_a_value)
    input_b_value = helper.find_value_by_name(g, original_matmul_node.input[1])
    input_b_shape = helper.get_shape_from_value_info(input_b_value)
    if len(input_a_shape) == 3:
        batch_count = input_a_shape[0]
    else:
        batch_count = input_a_shape[1]
    for i in range(batch_count):
        # Create Split nodes for input A
        starts_node = helper.list_to_constant(f"{input_a_value.name}_sliced_{i}_starts", (1, ), [i])
        ends_node = helper.list_to_constant(f"{input_a_value.name}_sliced_{i}_ends", (1, ), [i+1])
        axes_node = helper.list_to_constant(f"{input_a_value.name}_sliced_{i}_axes", (1, ), [len(input_a_shape) - 3])
        new_sliced_a_node = onnx.helper.make_node(
            'Slice',
            inputs = [input_a_value.name, starts_node.output[0], ends_node.output[0], axes_node.output[0]],
            outputs = [f"{input_a_value.name}_sliced_{i}"],
            name = f"{input_a_value.name}_sliced_{i}_for_{original_matmul_node.name}"
        )
        new_nodes.extend([starts_node, ends_node, axes_node, new_sliced_a_node])
        # Create Split nodes for input B
        starts_node = helper.list_to_constant(f"{input_b_value.name}_sliced_{i}_starts", (1, ), [i])
        ends_node = helper.list_to_constant(f"{input_b_value.name}_sliced_{i}_ends", (1, ), [i+1])
        axes_node = helper.list_to_constant(f"{input_b_value.name}_sliced_{i}_axes", (1, ), [len(input_b_shape) - 3])
        new_sliced_b_node = onnx.helper.make_node(
            'Slice',
            inputs = [input_b_value.name, starts_node.output[0], ends_node.output[0], axes_node.output[0]],
            outputs = [f"{input_b_value.name}_sliced_{i}"],
            name = f"{input_b_value.name}_sliced_{i}_for_{original_matmul_node.name}"
        )
        new_nodes.extend([starts_node, ends_node, axes_node, new_sliced_b_node])
        # Create MatMul nodes
        new_matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs = [new_sliced_a_node.output[0], new_sliced_b_node.output[0]],
            outputs = [f"{original_matmul_node.output[0]}_sliced_{i}"],
            name = f"{original_matmul_node.name}_sliced_{i}"
        )
        new_nodes.append(new_matmul_node)
        final_concat_inputs.append(new_matmul_node.output[0])
    # Create Concat nodes
    output_value = helper.find_value_by_name(g, original_matmul_node.output[0])
    if output_value is None:
        output_value = helper.find_output_by_name(g, original_matmul_node.output[0])
    if output_value is None:
        helper.logger.error(f"Cannot find value_info for {original_matmul_node.output[0]}")
    output_shape = helper.get_shape_from_value_info(output_value)
    new_concat_node = onnx.helper.make_node(
        "Concat",
        inputs = final_concat_inputs,
        outputs = [original_matmul_node.output[0]],
        name = f"{original_matmul_node.name}_final_concat",
        axis = len(output_shape) - 3
    )
    new_nodes.append(new_concat_node)
    # Add new nodes
    g.node.extend(new_nodes)
    # Delete original nodes
    g.node.remove(original_matmul_node)


def split_MatMul_Constant_input_then_concat(g, original_matmul_node):
    new_nodes = []
    final_concat_inputs = []
    # Get the batch count
    input_b_node = helper.find_node_by_output_name(g, original_matmul_node.input[1])
    input_b_np = helper.constant_to_numpy(input_b_node)
    if len(input_b_np.shape) == 3:
        batch_count = input_b_np.shape[0]
    else:
        batch_count = input_b_np.shape[1]
    for i in range(batch_count):
        # Create new constant node
        if len(input_b_np.shape) == 3:
            new_np = input_b_np[i:i+1, ...]
        else:
            new_np = input_b_np[:, i:i+1, ...]
        new_weight = helper.numpy_to_constant(f"{input_b_node.name}_sliced_{i}", new_np)
        new_nodes.append(new_weight)
        # Create MatMul nodes
        new_matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs = [original_matmul_node.input[0], new_weight.output[0]],
            outputs = [f"{original_matmul_node.output[0]}_sliced_{i}"],
            name = f"{original_matmul_node.name}_sliced_{i}"
        )
        new_nodes.append(new_matmul_node)
        final_concat_inputs.append(new_matmul_node.output[0])
    # Create Concat nodes
    output_value = helper.find_value_by_name(g, original_matmul_node.output[0])
    output_shape = helper.get_shape_from_value_info(output_value)
    new_concat_node = onnx.helper.make_node(
        "Concat",
        inputs = final_concat_inputs,
        outputs = [original_matmul_node.output[0]],
        name = f"{original_matmul_node.name}_final_concat",
        axis = len(output_shape) - 3
    )
    new_nodes.append(new_concat_node)
    # Add new nodes
    g.node.extend(new_nodes)
    # Delete original value info
    input_b_value = helper.find_value_by_name(g, original_matmul_node.input[1])
    if input_b_value is not None:
        g.value_info.remove(input_b_value)
    # Delete original nodes
    g.node.remove(original_matmul_node)
    g.node.remove(input_b_node)


def special_MatMul_process(g):
    for node in g.node:
        if node.op_type != 'MatMul':
            continue
        input_a_name = node.input[0]
        input_a_value = helper.find_value_by_name(g, input_a_name)
        input_b_name = node.input[1]
        input_b_value = helper.find_value_by_name(g, input_b_name)
        if input_a_value is None or input_b_value is None:
            continue
        input_a_shape = helper.get_shape_from_value_info(input_a_value)
        input_b_shape = helper.get_shape_from_value_info(input_b_value)
        # Check shapes and choose the process
        # Normal case, Skip
        if len(input_b_shape) == 2:
            continue
        # Too many dimensions or too few dimensions. Not supported. Skip
        if len(input_a_shape) > 4 or len(input_b_shape) > 4:
            helper.logger.warning(f"Cannot optimize MatMul {node.name}: inputs have too many dimensions.")
            continue
        if len(input_a_shape) < 2 or len(input_b_shape) < 2:
            helper.logger.warning(f"Cannot optimize MatMul {node.name}: inputs have two few dimensions.")
            continue
        # For 4 dimension, check the first dimension (should be 1) and treated as 3 dimensions.
        extra_dim = None
        if len(input_a_shape) == 4:
            extra_dim = input_a_shape[0]
            input_a_shape = input_a_shape[1:]
        if len(input_b_shape) == 4:
            if input_b_shape[0] != extra_dim:
                helper.logger.warning(f"Cannot optimize MatMul {node.name}: input dimension batch sizes does not match ({extra_dim} vs {input_b_shape[0]}).")
                continue
            input_b_shape = input_b_shape[1:]
        # Check input B dimension
        # If B is 1 x W x V, it is the same as normal case.
        if input_b_shape[0] == 1:
            continue
        # If B is B x W x V, but B is a constant.
        input_b_node = helper.find_node_by_output_name(g, input_b_name)
        if input_b_node is not None and input_b_node.op_type == 'Constant':
            # Constant input
            helper.logger.debug(f"Optimizing MatMul node {node.name}: split constant input.")
            split_MatMul_Constant_input_then_concat(g, node)
        # If B is B x W x V and A is 1 x H x W, do the swap.
        elif len(input_a_shape) == 2 or (input_a_shape[0] == 1 and (extra_dim is None or extra_dim == 1)):
            helper.logger.debug(f"Optimizing MatMul node {node.name}: swap input.")
            swap_MatMul_inputs(g, node)
        # If B is B x W x V and A is B x H x W, do the split.
        elif input_b_shape[0] == input_a_shape[0]:
            helper.logger.debug(f"Optimizing MatMul node {node.name}: split input batch.")
            split_MatMul_batch_then_concat(g, node)
        # Other cases are not supported: If B is B x W x V but A is X x H x W.
        else:
            helper.logger.warning(f"Cannot optimize MatMul {node.name}: unknown reason. Might be shape mismatch.")
            continue
    other.topological_sort(g)