"""Optimizations that replace one node with another.
"""
from os import dup
import struct
import copy
import logging
import onnx.helper
import numpy as np
from . import helper
from . import modhelper
from .other import topological_sort

def replace_initializer_with_Constant(g, duplicate_shared_weights=False):
    """
    Replace initializers with Constant and a corresponding value_info
    If the initializer has related input, remove it.

    :param g: the onnx graph
    """

    input_map = {i.name: i for i in g.input}
    for tensor in g.initializer:
        # Check for the initializer related input and remove it
        if tensor.name in input_map:
            value_info = input_map[tensor.name]
            g.input.remove(value_info)
        following_nodes = helper.find_nodes_by_input_name(g, tensor.name)
        if duplicate_shared_weights and len(following_nodes) >= 2:
            for i, node in enumerate(following_nodes):
                new_name = tensor.name + "_duplicated_No" + str(i) if i > 0 else tensor.name
                helper.logger.debug(f"Duplicating weight: {tensor.name} -> {new_name}")
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
        else:
            new_name = tensor.name
            new_node = onnx.helper.make_node(
                "Constant",
                [],
                [new_name],
                name=new_name,
                value=tensor
            )
            # Add node to lists
            g.node.extend([new_node])

        # if value info already exists, remove it as well.
        value_info = helper.find_value_by_name(g, tensor.name)
        if value_info is not None:
            g.value_info.remove(value_info)

    # Remove original initializer
    while len(g.initializer) != 0:
        g.initializer.pop()

    topological_sort(g)

def replace_Reshape_with_Flatten(g):
    """
    Replace Reshape node into Flatten node if applicable.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Reshape':
            continue
        found_Gemm = False
        # Flatten could be followed by Gemm
        for i in g.node:
            if len(i.input) == 0 or i.input[0] != node.output[0]:
                continue
            if i.op_type == 'Gemm':
                found_Gemm = True
                break
        # Check weight
        shape_node = helper.find_node_by_output_name(g, node.input[1])
        if shape_node.op_type != 'Constant':
            continue
        shape_value = helper.constant_to_numpy(shape_node)
        if (shape_value.size != 2 or shape_value[0] != 1) and not found_Gemm:
            continue
        # The first dimension must be the same
        input_value = helper.find_value_by_name(g, node.input[0])
        output_value = helper.find_value_by_name(g, node.output[0])
        if input_value is None or len(input_value.type.tensor_type.shape.dim) < 2:
            continue
        if output_value is None or len(output_value.type.tensor_type.shape.dim) != 2:
            continue
        if input_value.type.tensor_type.shape.dim[0].dim_value != output_value.type.tensor_type.shape.dim[0].dim_value:
            continue
        # Replace it
        node.op_type = "Flatten"
        for _ in range(len(node.attribute)):
            node.attribute.pop()
        shape_value = helper.find_value_by_name(g, shape_node.output[0])
        node.input.pop()
        if len(helper.find_following_nodes_by_input_value_name(g, shape_node.output[0])) <= 1:
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
            logging.warn("Cannot get shape for Squeeze")
            continue
        shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]
        if len(shape) == 0:
            g.value_info.remove(output_value)
            continue
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
            helper.logger.warn("Cannot get shape for Unsqueeze " + node.name )
            continue
        shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]
        if len(shape) == 0:
            g.value_info.remove(output_value)
            continue

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
            helper.logger.warn("Both strides and dilations are set in ", node.name)
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
        if input_value is None:
            input_value = helper.find_output_by_name(g, node.input[0])
        if input_value is None or len(input_value.type.tensor_type.shape.dim) == 0:
            continue
        # Check for case where dimension could be 0 or -1
        tmp = True
        for d in input_value.type.tensor_type.shape.dim:
            tmp = tmp and (d.dim_value > 0)
        if not tmp:
            continue
        # Repalce it
        input_shape = [
            int(d.dim_value) for d in input_value.type.tensor_type.shape.dim]
        node_name = node.output[0]
        new_node = helper.list_to_constant(
            node_name, [len(input_shape)], input_shape)
        g.node.extend([new_node])
        node_to_remove.append(node)
        helper.logger.debug(f"Shape node {node.name} is replaced by Constant.")

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

def replace_ConstantOfShape_with_constant(g):
    """Replace Shape with Constant.\\
    This is the first step of reshape constant folding.

    :param g: the input graph\\
    :return: if anything modified, return true.
    """
    node_to_remove = []
    for node in g.node:
        # Find a Shape
        if node.op_type != 'ConstantOfShape':
            continue
        # Check  input
        pre_node = helper.find_node_by_output_name(g, node.input[0])
        if pre_node is not None and pre_node.op_type != 'Constant':
            continue
        elif pre_node is not None:
            _, target_shape = helper.constant_to_list(pre_node)
        else:
            pre_initializer = helper.find_initializer_by_name(g, node.input[0])
            if pre_initializer is None:
                continue
            target_shape = helper.initializer_to_numpy(pre_initializer)

        helper.logger.debug(f"Replacing ConstantOfShape {node.name} to Constant.")
        # Get value to fill
        value_attr = helper.get_attribute_by_name(node, 'value')
        if value_attr is None:
            value = [0.0]
        else:
            value = helper.initializer_to_numpy(value_attr.t)

        node_name = node.output[0]
        new_node = helper.numpy_to_constant(node_name, np.full(target_shape, value[0]))
        g.node.extend([new_node])

        # remove old node
        node_to_remove.append(node)

        # delete value_info
        modhelper.delete_value_with_name_if_exists(g, node.input[0])

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
        if shape is None or len(shape) == 0:
            continue

        output_val_names = list(node.output)

        axis = 0
        split = []
        for item in node.attribute:
            if item.name == 'axis':
                axis = item.i
            if item.name == 'split':
                split = item.ints

        # For opset 11, axis could be negative.
        if axis < 0:
            axis = len(shape) + axis

        length = input_value.type.tensor_type.shape.dim[axis].dim_value
        if len(split) > 0:
            n_out = len(split)
            pos = 0
            for i in range(n_out):
                pos += split[i]
                new_node_name = output_val_names[i]
                # Construct starts, ends, axes
                starts_name = new_node_name + '_starts_' + str(i)
                ends_name = new_node_name + '_ends_' + str(i)
                axes_name = new_node_name + '_axes_' + str(i)
                starts_node = helper.list_to_constant(starts_name, (1, ), [int(pos-split[i])])
                ends_node = helper.list_to_constant(ends_name, (1, ), [int(pos)])
                axes_node = helper.list_to_constant(axes_name, (1, ), [int(axis)])
                # Construtc node
                new_node = onnx.helper.make_node(
                    op_type='Slice',
                    inputs=[node.input[0], starts_name, ends_name, axes_name],
                    outputs=[node.output[i]],
                    name=new_node_name
                )
                g.node.extend([starts_node, ends_node, axes_node, new_node])
            node_to_remove.append(node)
        else:
            n_out = len(output_val_names)
            width = length//n_out
            for i in range(n_out):
                new_node_name = output_val_names[i]
                # Construct starts, ends, axes
                starts_name = new_node_name + '_starts_' + str(i)
                ends_name = new_node_name + '_ends_' + str(i)
                axes_name = new_node_name + '_axes_' + str(i)
                starts_node = helper.list_to_constant(starts_name, (1, ), [int(i*width)])
                ends_node = helper.list_to_constant(ends_name, (1, ), [int((1+i)*width)])
                axes_node = helper.list_to_constant(axes_name, (1, ), [int(axis)])
                # Construtc node
                new_node = onnx.helper.make_node(
                    op_type='Slice',
                    inputs=[node.input[0], starts_name, ends_name, axes_name],
                    outputs=[node.output[i]],
                    name=new_node_name
                )
                g.node.extend([starts_node, ends_node, axes_node, new_node])
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

        prev_shape_value_info = helper.find_value_by_name(g, input_op_node_name)
        prev_shape_value_info = helper.find_input_by_name(g, input_op_node_name) if prev_shape_value_info is None else prev_shape_value_info
        if prev_shape_value_info is None:
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(prev_shape_value_info)
        scale_shape, scale_data = helper.constant_to_list(mul_value_node)

        # channel dimension
        c_dim = previous_node_output_shape[1] if len(previous_node_output_shape) > 1 else 1

        # only allow channelwise mul or const mul
        if scale_shape == [1, c_dim, 1, 1]:
            muls = scale_data
        elif scale_shape == [c_dim, 1, 1]:
            muls = scale_data
        elif scale_shape == 1:
            muls = scale_data * c_dim
        else:
            continue

        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
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

        modhelper.delete_value_with_name_if_exists(g, mul_value_node.output[0])

        g.node.extend([bn_node])
        g.node.extend([mean_value_node])
        g.node.extend([variance_value_node])
        g.node.extend([bias_value_node])
        g.node.extend([new_mul_value_node])

        node_to_del.extend([mul_op_node])
        if len(helper.find_following_nodes_by_input_value_name(g, mul_value_node.output[0])) <= 1:
            node_to_del.extend([mul_value_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

def replace_div_to_bn(g):
    """Replace single Div node with Batchnorm node.
    :param g: input graph.
    :return:
    """
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Div':
            continue

        div_op_node = node

        # only support one input node
        if len(div_op_node.input) != 2: # OP node and value node
            continue

        input_op_node_name = div_op_node.input[0]
        div_value_node = helper.find_node_by_output_name(g, div_op_node.input[1])
        if not div_value_node or div_value_node.op_type != 'Constant':
            continue

        prev_shape_value_info = helper.find_value_by_name(g, input_op_node_name)
        prev_shape_value_info = helper.find_input_by_name(g, input_op_node_name) if prev_shape_value_info is None else prev_shape_value_info
        if prev_shape_value_info is None:
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(prev_shape_value_info)
        scale_shape, scale_data = helper.constant_to_list(div_value_node)

        # channel dimension
        c_dim = previous_node_output_shape[1] if len(previous_node_output_shape) > 1 else 1

        # only allow channelwise div or const div
        if scale_shape == [1, c_dim, 1, 1]:
            muls = scale_data
        elif scale_shape == [c_dim, 1, 1]:
            muls = scale_data
        elif scale_shape == 1:
            muls = scale_data * c_dim
        else:
            continue

        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
        muls = [1 / i for i in muls]
        bn_name = div_op_node.output[0]
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
            [div_op_node.output[0]],
            name=bn_name,
            epsilon=0.00000001
        )

        modhelper.delete_value_with_name_if_exists(g, div_value_node.output[0])

        g.node.extend([bn_node])
        g.node.extend([mean_value_node])
        g.node.extend([variance_value_node])
        g.node.extend([bias_value_node])
        g.node.extend([new_mul_value_node])

        node_to_del.extend([div_op_node])
        if len(helper.find_following_nodes_by_input_value_name(g, div_value_node.output[0])) <= 1:
            node_to_del.extend([div_value_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)


def replace_add_to_bn(g):
    """Replace single Add node with Batchnorm node.
    :param g: input graph.
    :return:
    """
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Add':
            continue

        add_op_node = node

        # only support one input node
        if len(add_op_node.input) != 2: # OP node and value node
            continue

        input_op_node_name = add_op_node.input[0]
        add_value_node = helper.find_node_by_output_name(g, add_op_node.input[1])
        if not add_value_node or add_value_node.op_type != 'Constant':
            continue

        prev_shape_value_info = helper.find_value_by_name(g, input_op_node_name)
        prev_shape_value_info = helper.find_input_by_name(g, input_op_node_name) if prev_shape_value_info is None else prev_shape_value_info
        if prev_shape_value_info is None:
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(prev_shape_value_info)
        bias_shape, bias_data = helper.constant_to_list(add_value_node)

        # channel dimension
        c_dim = previous_node_output_shape[1] if len(previous_node_output_shape) > 1 else 1

        # only allow channelwise add or const add
        if bias_shape == [1, c_dim, 1, 1]:
            bias = bias_data
        elif bias_shape == [c_dim, 1, 1]:
            bias = bias_data
        elif bias_shape == 1:
            bias = bias_data * c_dim
        else:
            continue

        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
        bn_name = add_op_node.output[0]
        mean_value_node = helper.list_to_constant(bn_name+'_mean', np.array(zeros).shape, zeros)
        variance_value_node = helper.list_to_constant(bn_name+'_var', np.array(ones).shape, ones)
        scale_value_node = helper.list_to_constant(bn_name+'_mul', np.array(ones).shape, ones)
        new_add_value_node = helper.list_to_constant(bn_name+'_add', np.array(bias).shape, bias)

        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            [input_op_node_name,
            scale_value_node.output[0],
            new_add_value_node.output[0],
            mean_value_node.output[0],
            variance_value_node.output[0]],
            [add_op_node.output[0]],
            name=bn_name,
            epsilon=0.00000001
        )

        modhelper.delete_value_with_name_if_exists(g, add_value_node.output[0])

        g.node.extend([bn_node])
        g.node.extend([mean_value_node])
        g.node.extend([variance_value_node])
        g.node.extend([scale_value_node])
        g.node.extend([new_add_value_node])

        node_to_del.extend([add_op_node])
        if len(helper.find_following_nodes_by_input_value_name(g, add_value_node.output[0])) <= 1:
            node_to_del.extend([add_value_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

def replace_sub_to_bn(g):
    """Replace single Sub node with BatchNorm node.
    :param g: input graph.
    :return:
    """
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Sub':
            continue

        sub_op_node = node

        # only support one input node
        if len(sub_op_node.input) != 2: # OP node and value node
            continue

        # Check the input type
        input_1st_name = sub_op_node.input[0]
        input_2nd_name = sub_op_node.input[1]
        input_1st_node = helper.find_node_by_output_name(g, input_1st_name)
        input_2nd_node = helper.find_node_by_output_name(g, input_2nd_name)
        if input_1st_node is not None and input_1st_node.op_type == 'Constant':
            real_input_name = input_2nd_name
            reverse = True
            constant_node = input_1st_node
        elif input_2nd_node is not None and input_2nd_node.op_type == 'Constant':
            real_input_name = input_1st_name
            reverse = False
            constant_node = input_2nd_node
        else:
            continue

        # Get shapes
        prev_shape_value_info = helper.find_value_by_name(g, real_input_name)
        prev_shape_value_info = helper.find_input_by_name(g, real_input_name) if prev_shape_value_info is None else prev_shape_value_info
        if prev_shape_value_info is None:
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(prev_shape_value_info)
        bias_shape, bias_data = helper.constant_to_list(constant_node)

        # channel dimension
        c_dim = previous_node_output_shape[1] if len(previous_node_output_shape) > 1 else 1

        # only allow channelwise sub or const sub
        if bias_shape == [1, c_dim, 1, 1]:
            bias = bias_data
        elif bias_shape == [c_dim, 1, 1]:
            bias = bias_data
        elif bias_shape == 1:
            bias = bias_data * c_dim
        else:
            continue

        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
        # If reversed provide special scalar
        if reverse:
            scale = [-1.0] * c_dim
        else:
            scale = ones
            bias = [-1.0 * i for i in bias]
        bn_name = sub_op_node.output[0]
        mean_value_node = helper.list_to_constant(bn_name+'_mean', np.array(zeros).shape, zeros)
        variance_value_node = helper.list_to_constant(bn_name+'_var', np.array(ones).shape, ones)
        scale_value_node = helper.list_to_constant(bn_name+'_mul', np.array(scale).shape, scale)
        new_add_value_node = helper.list_to_constant(bn_name+'_add', np.array(bias).shape, bias)

        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            [real_input_name,
            scale_value_node.output[0],
            new_add_value_node.output[0],
            mean_value_node.output[0],
            variance_value_node.output[0]],
            [sub_op_node.output[0]],
            name=bn_name,
            epsilon=0.00000001
        )

        modhelper.delete_value_with_name_if_exists(g, constant_node.output[0])

        g.node.extend([bn_node])
        g.node.extend([mean_value_node])
        g.node.extend([variance_value_node])
        g.node.extend([scale_value_node])
        g.node.extend([new_add_value_node])

        node_to_del.extend([sub_op_node])
        if len(helper.find_following_nodes_by_input_value_name(g, constant_node.output[0])) <= 1:
            node_to_del.extend([constant_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

def replace_sub_with_bn_and_add(g):
    """Replace two input Sub node with BN and Add: A - B = A + (-1) * B
    :param g: input graph.
    :return:
    """
    for node in g.node:
        if node.op_type != 'Sub':
            continue

        sub_op_node = node

        # only support one input node
        if len(sub_op_node.input) != 2: # OP node and value node
            continue

        # Check the input type
        input_1st_name = sub_op_node.input[0]
        input_2nd_name = sub_op_node.input[1]
        input_1st_node = helper.find_node_by_output_name(g, input_1st_name)
        input_2nd_node = helper.find_node_by_output_name(g, input_2nd_name)
        if input_1st_node is not None and input_1st_node.op_type == 'Constant':
            continue
        elif input_2nd_node is not None and input_2nd_node.op_type == 'Constant':
            continue

        # Get shapes
        input_2nd_value_info = helper.find_value_by_name(g, input_2nd_name)
        if input_2nd_value_info is None:
            input_2nd_value_info = helper.find_input_by_name(g, input_2nd_name)
        if input_2nd_value_info is None:
            continue

        # Get channel dimension
        _ , input_2nd_shape = helper.find_size_shape_from_value(input_2nd_value_info)
        if len(input_2nd_shape) < 2:
            helper.logger.debug(f"{sub_op_node.name} cannot be replaced due to the input shape.")
            continue
        c_dim = input_2nd_shape[1]

        # Create * -1 bn node.
        ones = [1.0] * c_dim
        zeros = [0.0] * c_dim
        scale = [-1.0] * c_dim
        bn_name = input_2nd_name + '_neg_for_' + node.name
        mean_value_node = helper.list_to_constant(bn_name+'_mean', np.array(zeros).shape, zeros)
        variance_value_node = helper.list_to_constant(bn_name+'_var', np.array(ones).shape, ones)
        scale_value_node = helper.list_to_constant(bn_name+'_mul', np.array(scale).shape, scale)
        bias_value_node = helper.list_to_constant(bn_name+'_add', np.array(zeros).shape, zeros)
        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            [input_2nd_name,
            scale_value_node.output[0],
            bias_value_node.output[0],
            mean_value_node.output[0],
            variance_value_node.output[0]],
            [bn_name],
            name=bn_name,
            epsilon=0.00000001
        )

        # Change sub to add
        sub_op_node.op_type = "Add"
        # Replace add input
        modhelper.replace_node_input(sub_op_node, input_2nd_name, bn_name)

        g.node.extend([scale_value_node, bias_value_node, mean_value_node, variance_value_node, bn_node])

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


def replace_constant_input_concat_with_pad(g):
    """If single input is concating with constant node of same number. Replace it with pad. Currently only support 2-3 inputs.
    :param g: input graph.
    :return:
    """
    node_to_del = []
    for node in g.node:
        # Check for Concat node
        if node.op_type != 'Concat':
            continue

        # Check concat node input
        mode = None
        value = 0
        real_input_name = None
        if len(node.input) == 2:
            input_1st_node = helper.find_node_by_output_name(g, node.input[0])
            input_2nd_node = helper.find_node_by_output_name(g, node.input[1])
            if input_1st_node is not None and input_1st_node.op_type == 'Constant':
                mode = 'left'
                constant_value = helper.constant_to_numpy(input_1st_node)
                real_input_name = node.input[1]
                value = constant_value.flatten()[0].item()
                # Check if the values are all the same.
                if np.any(constant_value - value):
                    continue
            elif input_2nd_node is not None and input_2nd_node.op_type == 'Constant':
                mode = 'right'
                constant_value = helper.constant_to_numpy(input_2nd_node)
                real_input_name = node.input[0]
                value = constant_value.flatten()[0].item()
                # Check if the values are all the same.
                if np.any(constant_value - value):
                    continue
            else:
                # No constant input case
                continue
        elif len(node.input) == 3:
            # For 3 inputs concat node, the 1st and the 3rd input should be constant with the same value.
            input_1st_node = helper.find_node_by_output_name(g, node.input[0])
            input_2nd_node = helper.find_node_by_output_name(g, node.input[1])
            input_3rd_node = helper.find_node_by_output_name(g, node.input[2])
            if input_1st_node is None or input_1st_node.op_type != 'Constant' or \
                input_3rd_node is None or input_3rd_node.op_type != 'Constant':
                continue
            mode = 'both'
            real_input_name = node.input[1]
            input_1st_value = helper.constant_to_numpy(input_1st_node)
            input_3rd_value = helper.constant_to_numpy(input_3rd_node)
            value = input_1st_value.flatten()[0].item()
            # Check if all the values are all the same
            if np.any(input_1st_value - value):
                continue
            elif np.any(input_3rd_value - value):
                continue
        else:
            # Too many inputs case.
            continue
        # Make weight nodes
        input_value_info = helper.find_value_by_name(g, real_input_name)
        if input_value_info is None:
            continue
        input_shape = helper.get_shape_from_value_info(input_value_info)
        pads = [0] * (len(input_shape) * 2)
        axis = helper.get_var_attribute_by_name(node, 'axis', 'int')
        if axis < 0:
            axis = len(input_shape) - axis
        if mode == 'left':
            left_value_info = helper.find_value_by_name(g, node.input[0])
            left_input_shape = helper.get_shape_from_value_info(left_value_info)
            pads[axis] = left_input_shape[axis]
        elif mode == 'right':
            right_value_info = helper.find_value_by_name(g, node.input[1])
            right_input_shape = helper.get_shape_from_value_info(right_value_info)
            pads[axis + len(input_shape)] = right_input_shape[axis]
        else:
            # mode shoule be both
            left_value_info = helper.find_value_by_name(g, node.input[0])
            left_input_shape = helper.get_shape_from_value_info(left_value_info)
            pads[axis] = left_input_shape[axis]
            right_value_info = helper.find_value_by_name(g, node.input[2])
            right_input_shape = helper.get_shape_from_value_info(right_value_info)
            pads[axis + len(input_shape)] = right_input_shape[axis]
        pads_node = helper.list_to_constant(
            node.name + '_pads',
            (len(pads), ),
            pads
        )
        constant_value_node = helper.scalar_to_constant(
            node.name + '_constant_value',
            value
        )
        # Create new Pad node
        new_pad_node = onnx.helper.make_node(
            "Pad",
            [real_input_name, pads_node.name, constant_value_node.name],
            [node.output[0]],
            name = node.name,
            mode = "constant"
        )
        # Replace
        node_to_del.append(node)
        g.node.extend([pads_node, constant_value_node, new_pad_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)


def replace_Gather_with_Reshape(g):
    """
    Replace Gather nodes with Reshape node.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find Gather node
        if node.op_type != 'Gather':
            continue
        # Get the shape and Construct the shape
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            helper.logger.warn(f"Cannot get shape for Gather: {node.name}")
            continue
        shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]
        # Get the axis attribute
        axis = helper.get_var_attribute_by_name(node, 'axis', 'int')
        if axis is None:
            axis = 0
        # Get the indice node.
        indice_node = helper.find_node_by_output_name(g, node.input[1])
        if indice_node is None or indice_node.op_type != 'Constant':
            helper.logger.debug(f"Gather {node.name} indice input is not a constant node. Skip.")
            continue
        indice_np = helper.constant_to_numpy(indice_node)
        # Check if it is covering all the indices
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            input_value = helper.find_input_by_name(g, node.input[0])
        if input_value is None:
            helper.logger.warn(f"Cannot get shape for Gather: {node.name}")
            continue
        input_shape = helper.get_shape_from_value_info(input_value)
        if axis >= len(input_shape):
            continue
        if indice_np.flatten().size != input_shape[axis]:
            continue
        # Check if the indices are in order. Then it is just a normal reshape.
        prev = -1
        is_consecutive = True
        for i in indice_np.flatten():
            if i != prev + 1:
                is_consecutive = False
                break
            prev = i
        if is_consecutive:
            const_node = helper.list_to_constant(node.name + "_shape", [len(shape)], shape)
            new_node = onnx.helper.make_node(
                "Reshape",
                [node.input[0], node.name + "_shape"],
                node.output,
                name=node.name
            )
            g.node.extend([const_node, new_node])
            node_to_remove.append(node)
            continue
        # Check input shape can be turned into a reshape and a transpose
        is_tranposed = True
        if len(indice_np.shape) != 2:
            continue
        prev = -1
        for i in indice_np.transpose().flatten():
            if i != prev + 1:
                is_tranposed = False
                break
            prev = i
        if is_tranposed:
            # Create a reshape first.
            reshape_name = node.name + '_reshape_before_transpose'
            before_transpose_shape = copy.copy(shape)
            before_transpose_shape[axis] = shape[axis + 1]
            before_transpose_shape[axis + 1] = shape[axis]
            reshape_shape_node = helper.list_to_constant(node.name + "_shape_before_transpose", [len(shape)], before_transpose_shape)
            reshape_node = onnx.helper.make_node(
                "Reshape",
                [node.input[0], node.name + "_shape_before_transpose"],
                [reshape_name],
                name=reshape_name
            )
            # Create a transpose node.
            perm = [i for i in range(len(shape))]
            perm[axis] = axis + 1
            perm[axis + 1] = axis
            transpose_node = onnx.helper.make_node(
                "Transpose",
                [reshape_name],
                node.output,
                name=node.name,
                perm=perm
            )
            g.node.extend([reshape_shape_node, reshape_node, transpose_node])
            node_to_remove.append(node)
            continue
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)


def replace_Gather_with_Slice(g):
    """
    Replace Gather nodes with slice node.
    (Special process function for model_se)

    :param g: the input graph
    """
    node_to_remove = []
    axes_node = helper.list_to_constant('gather_nodes_axes', [1], [0])
    g.node.append(axes_node)
    for node in g.node:
        # Find Gather node
        if node.op_type != 'Gather':
            continue
        # Get the shape and Construct the shape
        constant_input = helper.find_node_by_output_name(g, node.input[1])
        if constant_input.op_type != 'Constant':
            logging.warning(f"Unsupported Gather: {node.name}")
            continue
        shape, constant_value = helper.constant_to_list(constant_input)
        if isinstance(shape, int):
            continue
        elif len(shape)> 1 or shape[0] ==1:
            continue
        constant_value = constant_value[0]
        starts_node = helper.list_to_constant(node.name + "_starts", [1], [constant_value])
        ends_node = helper.list_to_constant(node.name + "_ends", [1], [constant_value + 1])
        new_node = onnx.helper.make_node(
            "Slice",
            [node.input[0], starts_node.output[0], ends_node.output[0], axes_node.output[0]],
            [node.output[0]],
            name=node.name
        )
        g.node.extend([starts_node, ends_node, new_node])
        node_to_remove.append(node)
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)


def replace_Expand_with_Reshape(g):
    """
    Replace Expand nodes with Reshape node.

    :param g: the input graph
    """
    node_to_remove = []
    for node in g.node:
        # Find Gather node
        if node.op_type != 'Expand':
            continue
        # Check node input[1]
        input_shape_node = helper.find_node_by_output_name(g, node.input[1])
        if input_shape_node is None or input_shape_node.op_type != 'Constant':
            continue
        # Get the output shape and Construct the shape
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            helper.logger.warning(f"Cannot get output shape for Expand: {node.name}")
            continue
        output_shape = [dim.dim_value for dim in output_value.type.tensor_type.shape.dim]
        # Get input shape
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            input_value = helper.find_input_by_name(g, node.input[0])
        if input_value is None:
            helper.logger.error(f"Cannot get input shape for Expand: {node.name}")
            exit(1)
        input_shape = [dim.dim_value for dim in input_value.type.tensor_type.shape.dim]
        # Check if this is reshape.
        output_total_count = 1
        for i in output_shape:
            output_total_count *= i
        input_total_count = 1
        for i in input_shape:
            input_total_count *= i
        if input_total_count != output_total_count:
            continue
        # Construct new constant node
        new_shape = helper.list_to_constant(node.input[1], [len(output_shape)], output_shape)
        # Construct new reshape node.
        new_node = onnx.helper.make_node(
            "Reshape",
            [node.input[0], node.input[1]],
            node.output,
            name=node.name
        )
        g.node.extend([new_shape, new_node])
        node_to_remove.append(node)
        node_to_remove.append(input_shape_node)
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)
