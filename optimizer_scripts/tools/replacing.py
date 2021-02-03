"""Optimizations that replace one node with another.
"""
import struct
import copy
import logging
import onnx.helper
import numpy as np
from . import helper
from . import modhelper
from .other import topological_sort

def replace_initializer_with_Constant(g):
    """
    Replace initializers with Constant and a corresponding value_info

    :param g: the onnx graph
    """
    # Creat a set of existed node names
    node_names = set()
    for node in g.node:
        node_names.add(node.name)
    # Unused initializers should be removed
    unused_initializer = set()
    for tensor in g.initializer:
        unused_initializer.add(tensor.name)
    for node in g.node:
        for in_value in node.input:
            if in_value in unused_initializer:
                unused_initializer.remove(in_value)

    input_map = {i.name: i for i in g.input}
    for tensor in g.initializer:
        if tensor.name in unused_initializer:
            value_info = input_map[tensor.name]
            g.input.remove(value_info)
            continue
        # Convert init to a constant node
        if tensor.name not in node_names:
            new_name = tensor.name
        else:
            new_name = tensor.name + '_2'
            following_nodes = helper.find_nodes_by_input_name(g, tensor.name)
            for node in following_nodes:
                modhelper.replace_node_input(node, tensor.name, new_name)
        new_node = onnx.helper.make_node(
            "Constant",
            [],
            [new_name],
            name=new_name,
            value=tensor
        )
        # Add node to lists
        g.node.extend([new_node])
        # Add value info to lists
        value_info = input_map[tensor.name]
        g.value_info.extend([value_info])
        # Remove original input value info
        g.input.remove(value_info)
        # if value info exists, remove it as well.
        value_info = helper.find_value_by_name(g, tensor.name)
        if value_info is not None:
            g.value_info.remove(value_info)
    # Remove original initializer
    while len(g.initializer) != 0:
        g.initializer.pop()

def replace_Reshape_with_Flatten(g):
    """
    Replace Reshape node into Flatten node if applicable.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Reshape':
            continue
        found = False
        # Flatten must be followed by Gemm
        for i in g.node:
            if len(i.input) == 0 or i.input[0] != node.output[0]:
                continue
            if i.op_type == 'Gemm':
                found = True
                break
        if not found:
            continue
        shape_node = helper.find_node_by_output_name(g, node.input[1])
        if shape_node.op_type != 'Constant':
            continue
        # Replace it
        node.op_type = "Flatten"
        for _ in range(len(node.attribute)):
            node.attribute.pop()
        shape_value = helper.find_value_by_name(g, shape_node.output[0])
        node.input.pop()
        node_to_remove.append(shape_node)
        # If found shape value_info, remove it
        if shape_value != None:
            g.value_info.remove(shape_value)

    for node in node_to_remove:
        g.node.remove(node)

def replace_Squeeze_with_Reshape(g):
    """
    Replace Squeeze nodes with Reshape node.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find Squeeze node
        if node.op_type != 'Squeeze':
            continue
        # Get the shape and Construct the shape
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            raise RuntimeError("Cannot get shape for Squeeze")
        shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]
        const_node = helper.list_to_constant(node.name + "_shape", [len(shape)], shape)
        # Construct the Reshape layer with same input, output and name.
        new_node = onnx.helper.make_node(
            "Reshape",
            [node.input[0], node.name + "_shape"],
            node.output,
            name=node.name
        )
        # Append constructed nodes and append old node to remove_list
        g.node.extend([const_node, new_node])
        node_to_remove.append(node)
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)

def replace_Unsqueeze_with_Reshape(g):
    """
    Replace Unsqueeze nodes with Reshape node.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find Squeeze node
        if node.op_type != 'Unsqueeze':
            continue
        # Get the shape and Construct the shape
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            raise RuntimeError("Cannot get shape for Unsqueeze")
        shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]

        const_node = helper.list_to_constant(node.name + "_shape", [len(shape)], shape)
        # Construct the Reshape layer with same input, output and name.
        new_node = onnx.helper.make_node(
            "Reshape",
            [node.input[0], node.name + "_shape"],
            node.output,
            name=node.name
        )
        # Append constructed nodes and append old node to remove_list
        g.node.extend([const_node, new_node])
        node_to_remove.append(node)
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)

def replace_average_pool_with_GAP(g):
    """
    Replace AveragePool nodes with GlobalAveragePool node when available.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find a average pool layer
        if node.op_type != 'AveragePool':
            continue
        # Check attributes
        not_replace = False
        for attr in node.attribute:
            if attr.name == 'pads':
                if list(attr.ints) != [0, 0, 0, 0]:
                    not_replace = True
                    break
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
                value_info = helper.find_value_by_name(g, node.input[0])
                if value_info is None:
                    not_replace = True
                    break
                input_shape = []
                for dim in value_info.type.tensor_type.shape.dim:
                    input_shape.append(dim.dim_value)
                if input_shape[-2:] != kernel_shape:
                    not_replace = True
                    break
        if not_replace:
            continue
        # Replace it with GlobalAveragePool
        new_node = onnx.helper.make_node(
            "GlobalAveragePool",
            node.input,
            node.output,
            name=node.name
        )
        g.node.extend([new_node])
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)

def replace_dilated_conv(g):
    """
    If the dilation of a convolution is not (1, 1), replace it with a regular
    convolution with an expanded kernel.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Check if this is a conv layer
        if node.op_type != 'Conv':
            continue
        # Check if this has dilation
        has_dilations = False
        has_strides = False
        for attr in node.attribute:
            if attr.name == "dilations":
                dilations = list(attr.ints)
                if dilations != [1, 1]:
                    has_dilations = True
            if attr.name == "strides":
                strides = list(attr.ints)
                if strides != [1, 1]:
                    has_strides = True
        if has_dilations and has_strides:
            print("Warning: Both strides and dilations are set in ", node.name)
            continue
        if not has_dilations:
            continue
        # Construct new kernel
        w_node = helper.find_node_by_output_name(g, node.input[1])
        w_output = helper.find_value_by_name(g, node.input[1])
        shape = list(w_node.attribute[0].t.dims)
        # get original weight from float_data or raw data
        weight = list(w_node.attribute[0].t.float_data)
        if len(weight) == 0:
            # Unpack from raw data
            raw_data = w_node.attribute[0].t.raw_data
            weight = [i[0] for i in struct.iter_unpack('f', raw_data)]
        weight = np.array(weight)
        weight = np.reshape(weight ,shape)
        new_shape = copy.copy(shape)
        new_shape[2] = 1 + (shape[2] - 1) * dilations[0]
        new_shape[3] = 1 + (shape[3] - 1) * dilations[1]
        new_weight = np.zeros(new_shape)
        for batch in range(shape[0]):
            for ch in range(shape[1]):
                for h in range(shape[2]):
                    nh = h * dilations[0]
                    for w in range(shape[3]):
                        nw = w * dilations[1]
                        new_weight[batch, ch, nh, nw] = weight[batch, ch, h, w]
        tensor = onnx.helper.make_tensor(
            w_node.attribute[0].t.name,
            w_node.attribute[0].t.data_type,
            new_shape,
            new_weight.ravel()
        )
        new_w_node = onnx.helper.make_node(
            "Constant",
            [],
            list(w_node.output),
            name=w_node.name,
            value=tensor
        )
        g.node.extend([new_w_node])
        node_to_remove.append(w_node)
        # Modify attributes and value info shapes
        w_output.type.tensor_type.shape.dim[2].dim_value = new_shape[2]
        w_output.type.tensor_type.shape.dim[3].dim_value = new_shape[3]
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                attr.ints[0] = new_shape[2]
                attr.ints[1] = new_shape[3]
            if attr.name == "dilations":
                attr.ints[0] = 1
                attr.ints[1] = 1
    # Remove old weight nodes
    for node in node_to_remove:
        g.node.remove(node)

def replace_depthwise_1x1_with_bn(g):
    """Replace 1x1 DepthwiseConv node into BN node if applicable.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        # Check op_type
        if node.op_type != 'Conv':
            continue
        # Check attributes
        attr_map = {attr.name: attr for attr in node.attribute}
        if "group" not in attr_map or attr_map["group"].i == 1:
            continue
        if attr_map["kernel_shape"].ints[0] != 1 or attr_map["kernel_shape"].ints[1] != 1:
            continue
        if "pads" in attr_map and sum(attr_map["pads"].ints) != 0:
            continue
        # Check scale
        scale_node = helper.find_node_by_output_name(g, node.input[1])
        if scale_node is None or scale_node.attribute[0].t.dims[1] != 1:
            continue
        scale_node.attribute[0].t.dims.pop()
        scale_node.attribute[0].t.dims.pop()
        scale_node.attribute[0].t.dims.pop()
        scale_info = helper.find_value_by_name(g, node.input[1])
        if scale_info is not None:
            scale_info.type.tensor_type.shape.dim.pop()
            scale_info.type.tensor_type.shape.dim.pop()
            scale_info.type.tensor_type.shape.dim.pop()
        # Check bias
        if len(node.input) == 3:
            bias_name = node.input[2]
        else:
            bias_name = node.name + "_bias"
            bias_node = helper.list_to_constant(bias_name, [attr_map["group"].i], [0.0] * attr_map["group"].i)
            g.node.extend([bias_node])
        # Construct mean and vars
        mean_name = node.name + "_mean"
        mean_node = helper.list_to_constant(mean_name, [attr_map["group"].i], [0.0] * attr_map["group"].i)
        var_name = node.name + "_var"
        var_node = helper.list_to_constant(var_name, [attr_map["group"].i], [1.0] * attr_map["group"].i)
        g.node.extend([mean_node, var_node])
        # Convert
        bn_node = onnx.helper.make_node(
            op_type='BatchNormalization',
            inputs=[node.input[0], node.input[1], bias_name, mean_name, var_name],
            outputs=node.output,
            name=node.name,
            epsilon=0.00001,
            momentum=0.9
            )
        g.node.extend([bn_node])
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)

def replace_shape_with_constant(g):
    """Replace Shape with Constant.\\
    This is the first step of reshape constant folding.

    :param g: the input graph\\
    :return: if anything modified, return true.
    """
    node_to_remove = []
    for node in g.node:
        # Find a Shape
        if node.op_type != 'Shape':
            continue
        # Check its input
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            input_value = helper.find_input_by_name(g, node.input[0])
        if input_value is None or len(input_value.type.tensor_type.shape.dim) == 0:
            continue
        # Repalce it
        input_shape = [
            d.dim_value for d in input_value.type.tensor_type.shape.dim]
        node_name = node.output[0]
        new_node = helper.list_to_constant(
            node_name, [len(input_shape)], input_shape)
        g.node.extend([new_node])
        node_to_remove.append(node)

        # if the input value_info is not used by other node
        # delete this input value_info
        val_info_used = sum([input_value.name in node.input for node in g.node])
        if val_info_used == 1:
            g.value_info.remove(input_value)

    replaced = True if len(node_to_remove) > 0 else False

    for node in node_to_remove:
        g.node.remove(node)

    topological_sort(g)

    return replaced

def replace_split_with_slices(g):
    """Replace split node with slice nodes.
    :param g: input graph.
    :return:
    """
    node_to_remove = []
    for node in g.node:
        # Find a Split
        if node.op_type != 'Split':
            continue

        input_value = helper.find_value_by_name(g, node.input[0])
        if not input_value:
            input_value = helper.find_input_by_name(g, node.input[0])
        _, shape = helper.find_size_shape_from_value(input_value)
        if len(shape) == 0:
            continue

        output_val_names = list(node.output)

        axis = 0
        split = []
        for item in node.attribute:
            if item.name == 'axis':
                axis = item.i
            if item.name == 'split':
                split = item.ints

        length = input_value.type.tensor_type.shape.dim[axis].dim_value

        outputs = node.output
        if split is not []:
            n_out = len(node.attribute[1].ints)
            pos = 0
            for i in range(n_out):
                pos += node.attribute[1].ints[i]
                new_node_name = output_val_names[i]
                new_node = onnx.helper.make_node(
                    op_type='Slice',
                    inputs=[node.input[0]],
                    outputs=[new_node_name],
                    name=new_node_name,
                    axes=[axis],
                    ends=[pos],
                    starts=[pos-node.attribute[1].ints[i]]
                )
                g.node.extend([new_node])
            node_to_remove.append(node)
        else:
            n_out = len(outputs)
            width = length//n_out
            for i in range(n_out):
                new_node = onnx.helper.make_node(
                    op_type='Slice',
                    inputs=[node.input[0]],
                    outputs=[outputs[i]],
                    name=outputs[i],
                    axes=[axis],
                    ends=[(1+i)*width],
                    starts=[i*width]
                )
                g.node.extend([new_node])
            node_to_remove.append(node)

    for old_node in node_to_remove:
        g.node.remove(old_node)
    topological_sort(g)


def replace_ReduceMean_with_GlobalAveragePool(g):
    """
    Replace ReduceMean with GlobalAveragePool node when available.

    If there is preceeded Transpose, check the Transpose and the ReduceMean
    together. If the keep_dims is set to 0, add a Flatten.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find a ReduceMean layer
        if node.op_type != 'ReduceMean':
            continue
        # Find if it have previous Transpose and its attribute meet the need.
        prev_node = helper.find_node_by_output_name(g, node.input[0])
        if prev_node is not None and prev_node.op_type != 'Transpose':
            prev_node = None
        if prev_node is not None:
            perm = helper.get_list_attribute_by_name(prev_node, 'perm', 'int')
            if perm != [0, 2, 3, 1]:
                prev_node = None
        # Check attributes
        axes = helper.get_list_attribute_by_name(node, 'axes', 'int')
        keepdims = helper.get_var_attribute_by_name(node, 'keepdims', 'int')
        if axes is None:
            continue
        if prev_node is None and axes != [2, 3]:
            continue
        if prev_node is not None and axes != [1, 2]:
            continue
        if keepdims is None:
            keepdims = 1
        # Replace it with GlobalAveragePool
        if prev_node:
            input_list = prev_node.input
        else:
            input_list = node.input
        if keepdims == 1:
            output_list = node.output
        else:
            output_list = [node.output[0] + '_before_flatten']
            flatten_node = onnx.helper.make_node(
                "Flatten",
                output_list,
                node.output,
                name = node.name + "_flatten",
                axis = 1
            )
            g.node.extend([flatten_node])
        new_node = onnx.helper.make_node(
            "GlobalAveragePool",
            input_list,
            output_list,
            name=node.name
        )
        g.node.extend([new_node])
        node_to_remove.append(node)
        if prev_node:
            value = helper.find_value_by_name(g, prev_node.output[0])
            if value:
                g.value_info.remove(value)
            node_to_remove.append(prev_node)
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)

def replace_mul_to_bn(g):
    """Replace single Mul node with Batchnorm node.
    :param g: input graph.
    :return:
    """
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Mul':
            continue

        mul_op_node = node

        # only support one input node
        if len(mul_op_node.input) != 2: # OP node and value node
            continue

        input_op_node_name = mul_op_node.input[0]
        mul_value_node = helper.find_node_by_output_name(g, mul_op_node.input[1])
        if not mul_value_node or mul_value_node.op_type != 'Constant':
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(helper.find_value_by_name(g, input_op_node_name))
        scale_shape, scale_data = helper.constant_to_list(mul_value_node)


        # only allow 4 dim data input due to the hardware limitation
        if len(previous_node_output_shape) != 4:
            continue

        # channel dimension
        c_dim = previous_node_output_shape[1]

        # only allow channelwise mul or const mul
        if scale_shape != [1, c_dim, 1, 1] and scale_shape != 1:
            continue

        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
        muls = scale_data * c_dim
        bn_name = mul_op_node.output[0]
        mean_value_node = helper.list_to_constant(bn_name+'_mean', np.array(zeros).shape, zeros)
        variance_value_node = helper.list_to_constant(bn_name+'_var', np.array(ones).shape, ones)
        bias_value_node = helper.list_to_constant(bn_name+'_add', np.array(zeros).shape, zeros)
        new_mul_value_node = helper.list_to_constant(bn_name+'_mul', np.array(muls).shape, muls)

        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            [input_op_node_name,
            new_mul_value_node.output[0],
            bias_value_node.output[0],
            mean_value_node.output[0],
            variance_value_node.output[0]],
            [mul_op_node.output[0]],
            name=bn_name,
            epsilon=0.00000001
        )

        mid_val_info = helper.find_value_by_name(g, mul_op_node.output[0])
        scale_val_info = helper.find_value_by_name(g, mul_value_node.output[0])
        g.value_info.remove(mid_val_info)
        g.value_info.remove(scale_val_info)

        g.node.extend([bn_node])
        g.node.extend([mean_value_node])
        g.node.extend([variance_value_node])
        g.node.extend([bias_value_node])
        g.node.extend([new_mul_value_node])

        node_to_del.extend([mul_op_node])
        node_to_del.extend([mul_value_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

def replace_Sum_with_Adds(g):
    node_to_del = []

    for node in g.node:
        # Check for sum
        if node.op_type != 'Sum':
            continue
        # Check for input number
        if len(node.input) == 1:
            # If input number is 1, delete the sum node.
            following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
            for following_node in following_nodes:
                modhelper.replace_node_input(following_node, node.output[0], node.input[0])
            node_to_del.append(node)
            if helper.find_value_by_name(node.output[0]) is not None:
                g.value_info.remove(helper.find_value_by_name(node.output[0]))
        elif len(node.input) == 2:
            # If input number is 2, replace it with add.
            node.op_type = 'Add'
            continue
        elif len(node.input) > 2:
            # If input number is larger than 2, replace it with n-1 add.
            input_count = len(node.input)
            # First node has 2 inputs
            first_node = onnx.helper.make_node(
                "Add",
                [node.input[0], node.input[1]],
                [node.output[0] + '_replacement_1'],
                name=node.name + '_replacement_1'
            )
            # Last node has the same output as the original sum node
            last_node = onnx.helper.make_node(
                "Add",
                [node.output[0] + '_replacement_' + str(input_count - 2), node.input[input_count - 1]],
                [node.output[0]],
                name=node.name
            )
            g.node.extend([first_node, last_node])
            for i in range(2, input_count - 1):
                new_node = onnx.helper.make_node(
                    "Add",
                    [node.output[0] + '_replacement_' + str(i - 1), node.input[i]],
                    [node.output[0] + '_replacement_' + str(i)],
                    name=node.name + '_replacement_' + str(i)
                )
                g.node.extend([new_node])
            node_to_del.append(node)
        else:
            logging.error("Sum node must have at least 1 input.")
            quit(1)

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

