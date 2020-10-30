"""Optimization functions that are not fusing, eliminating or replacing. In most
cases, these are the modifications on the original nodes.
"""
import struct
import collections
import numpy as np
import onnx.helper
import math
import logging
from . import helper
from .modhelper import replace_node_input

def format_value_info_shape(g):
    """
    Replace -1 batch size in value info

    :param g: the onnx graph
    """
    for value in g.input:
        if len(value.type.tensor_type.shape.dim) > 0 and\
           (value.type.tensor_type.shape.dim[0].dim_value <= 0 or\
           not isinstance(value.type.tensor_type.shape.dim[0].dim_value, int)):
            value.type.tensor_type.shape.dim[0].dim_value = 1
    for value in g.output:
        if len(value.type.tensor_type.shape.dim) > 0 and\
           (value.type.tensor_type.shape.dim[0].dim_value <= 0 or\
           not isinstance(value.type.tensor_type.shape.dim[0].dim_value, int)):
            value.type.tensor_type.shape.dim[0].dim_value = 1
    for value in g.value_info:
        if len(value.type.tensor_type.shape.dim) > 0 and\
           (value.type.tensor_type.shape.dim[0].dim_value <= 0 or\
           not isinstance(value.type.tensor_type.shape.dim[0].dim_value, int)):
            value.type.tensor_type.shape.dim[0].dim_value = 1

def add_name_to_node(g):
    """
    If no name presents, give a name based on output name.

    :param g: the onnx graph
    """
    for node in g.node:
        if len(node.name) == 0:
            node.name = node.output[0]

def add_output_to_value_info(g):
    """
    If output does not present in value_info, copy one

    :param g: the onnx graph
    """
    for output in g.output:
        if helper.find_value_by_name(g, output.name) is None:
            g.value_info.extend([output])

def remove_nodes(g, cut_nodes=[], cut_types=[]):
    node_to_delete = []
    #Find target nodes
    for node in g.node:
        if node.name not in cut_nodes and node.op_type not in cut_types:
            continue
        else:
            node_to_delete.append(node)
    #Remove them and add new outputs
    new_output = []
    while node_to_delete:
        node = node_to_delete.pop()
        for input_name in node.input:
            value = helper.find_value_by_name(g, input_name)
            if value is not None and helper.find_output_by_name(g, input_name) is None:
                new_output.append(value)
        g.node.remove(node)
    g.output.extend(new_output)
    #Remove unreachable nodes
    visited_values = set()
    unused_constant_map = {}
    for input_value in g.input:
        visited_values.add(input_value.name)
    for node in g.node:
        if node.op_type == 'Constant':
            visited_values.add(node.output[0])
            unused_constant_map[node.output[0]] = node
            continue
        can_reach = True
        for input_name in node.input:
            if input_name not in visited_values:
                can_reach = False
                break
        if can_reach:
            for output_name in node.output:
                visited_values.add(output_name)
        else:
            node_to_delete.append(node)
    new_output = []
    while node_to_delete:
        node = node_to_delete.pop()
        for input_name in node.input:
            value = helper.find_value_by_name(g, input_name)
            if value is not None and helper.find_output_by_name(g, input_name) is None:
                new_output.append(value)
        g.node.remove(node)
    g.output.extend(new_output)
    #Remove unused constants
    for node in g.node:
        for input_name in node.input:
            if input_name in unused_constant_map:
                del unused_constant_map[input_name]
    for node in unused_constant_map.values():
        g.node.remove(node)
    #Remove unreachable value infos and outputs
    reachable_values = set()
    for input_value in g.input:
        reachable_values.add(input_value.name)
    for node in g.node:
        for input_name in node.input:
            reachable_values.add(input_name)
        for output_name in node.output:
            reachable_values.add(output_name)
    value_to_remove = []
    for value_info in g.value_info:
        if value_info.name not in reachable_values:
            value_to_remove.append(value_info)
    while value_to_remove:
        value_info = value_to_remove.pop()
        g.value_info.remove(value_info)
    for value_info in g.output:
        if value_info.name not in reachable_values:
            value_to_remove.append(value_info)
    while value_to_remove:
        value_info = value_to_remove.pop()
        g.output.remove(value_info)

def transpose_B_in_Gemm(g):
    """
    If transB is set in Gemm, transpose it

    :param g: the onnx graph
    """
    for node in g.node:
        if node.op_type != 'Gemm':
            continue
        do_it = False
        for attr in node.attribute:
            if attr.name == "transB":
                if attr.i == 1:
                    attr.i = 0
                    do_it = True
                    break
        if not do_it:
            continue
        # Transpose the weight and its output value
        w_node = helper.find_node_by_output_name(g, node.input[1])
        w_output = helper.find_value_by_name(g, node.input[1])
        dim_0 = w_output.type.tensor_type.shape.dim[0].dim_value
        dim_1 = w_output.type.tensor_type.shape.dim[1].dim_value
        w_output.type.tensor_type.shape.dim[0].dim_value = dim_1
        w_output.type.tensor_type.shape.dim[1].dim_value = dim_0
        w_node.attribute[0].t.dims[0] = dim_1
        w_node.attribute[0].t.dims[1] = dim_0
        if w_node.attribute[0].t.raw_data:
            raw_data = w_node.attribute[0].t.raw_data
            fl_data = [i[0] for i in struct.iter_unpack('f', raw_data)]
        else:
            fl_data = w_node.attribute[0].t.float_data
        w = np.reshape(fl_data, (dim_0, dim_1))
        w = w.transpose((1, 0)).flatten()
        if w_node.attribute[0].t.raw_data:
            buf = struct.pack('%sf' % len(w), *w)
            w_node.attribute[0].t.raw_data = buf
        else:
            for i in range(len(fl_data)):
                w_node.attribute[0].t.float_data[i] = w[i]

def topological_sort(g):
    """
    Topological sort all the layers.
    Assume a node do not take the same value as more than one inputs.

    :param g: the onnx graph
    """
    # TODO: Topological sort on the same branch
    # Map from node name to its input degree
    in_degree = {}
    # Map from value info name to the nodes using it as input
    output_nodes = collections.defaultdict(list)
    # Map from node name to node object
    node_map = {}
    to_add = collections.deque()
    # init
    length = len(g.node)
    for _ in range(length):
        node = g.node.pop()
        node_map[node.name] = node
        if len(node.input) == 0:
            to_add.append(node.name)
        else:
            in_degree[node.name] = len(node.input)
            for input_name in node.input:
                output_nodes[input_name].append(node.name)
    # sort
    # deal with input first
    for value_info in g.input:
        input_name = value_info.name
        for node_name in output_nodes[input_name]:
            in_degree[node_name] -= 1
            if in_degree[node_name] == 0:
                to_add.append(node_name)
                del in_degree[node_name]
    # main sort loop
    sorted_nodes = []
    while to_add:
        node_name = to_add.pop()
        node = node_map[node_name]
        del node_map[node_name]
        sorted_nodes.append(node)
        # Expect only one output name for each node
        next_node_names = []
        for output_name in node.output:
            next_node_names.extend(output_nodes[output_name])
        for next_node_name in next_node_names:
            in_degree[next_node_name] -= 1
            if in_degree[next_node_name] == 0:
                to_add.append(next_node_name)
                del in_degree[next_node_name]
    g.node.extend(sorted_nodes)
    if in_degree:
        raise RuntimeError("Unreachable nodes exist: {}".format(in_degree.keys()))
    if node_map:
        raise RuntimeError("Unused nodes exist: {}".format(node_map.keys()))


def inference_shapes(m):
    g = m.graph
    inferencing_shapes = True
    while inferencing_shapes:
        inferencing_shapes = False
        if inference_cov_shape(g):
            inferencing_shapes = True
        if inference_upsample_shape(g):
            inferencing_shapes = True
        if inference_split_shape(g):
            inferencing_shapes = True
        if inferencing_shapes:
            topological_sort(g)
            m = onnx.utils.polish_model(m)
            g = m.graph
    return m


def inference_upsample_shape(g):
    """For onnx v1.4.1+, onnx cannot inference upsample output shape. Let's\\
    do it ourselves. This function only inference the next upsample without\\
    output shape each time.

    :param g: the graph\\
    :return: True if any Upsample shape is generated. Otherwise, False.
    """
    for node in g.node:
        if node.op_type != 'Upsample':
            continue
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value and helper.get_shape_from_value_info(output_value):
            continue
        # Get input shape
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            continue
            #raise RuntimeError("Shape for {} has not been generated.".format(node.input[0]))
        if not helper.get_shape_from_value_info(input_value):
            continue
            #raise RuntimeError("Shape for {} is empty.".format(node.input[0]))
        input_shape = helper.get_shape_from_value_info(input_value)
        # Get upsample weight
        weight_node = helper.find_node_by_output_name(g, node.input[1])
        weight_shape, weight = helper.constant_to_list(weight_node)
        if len(input_shape) != weight_shape[0]:
            raise RuntimeError("Unmatch input shape and weight shape: {} vs {}".format(input_shape, weight_shape))
        # Calculate shape
        output_shape = list(input_shape)
        for i in range(len(output_shape)):
            output_shape[i] = int(input_shape[i] * weight[i])
        output_value = onnx.helper.make_tensor_value_info(
                node.output[0],
                input_value.type.tensor_type.elem_type,
                output_shape)
        g.value_info.extend([output_value])
        return True
    return False

def inference_cov_shape(g):
    processed = False
    for node in g.node:
        # Check for Conv output shape need to be inferrenced.
        if node.op_type != 'Conv':
            continue
        # Input shape is not ready yet. Skip.
        input_value_info = helper.find_value_by_name(g, node.input[0])
        if not input_value_info:
            input_value_info = helper.find_input_by_name(g, node.input[0])
        if not input_value_info:
            continue
        _, input_shape = helper.find_size_shape_from_value(input_value_info)
        if not input_shape:
            continue
        # Output shape is already there. Skip.
        output_value_info = helper.find_value_by_name(g, node.output[0])
        if not output_value_info:
            output_value_info = helper.find_output_by_name(g, node.output[0])
        if output_value_info and \
            helper.get_shape_from_value_info(output_value_info):
            continue

        # Now start the inference.
        # If auto_pad is set, use the auto_pad.
        auto_pad = helper.get_var_attribute_by_name(node, 'auto_pad', 'string')
        pads = None
        if auto_pad is not None and auto_pad != 'NOTSET':
            if auto_pad == 'SAME_LOWER' or auto_pad == 'SAME_UPPER':
                new_output_value_info = onnx.helper.make_tensor_value_info(
                    node.output[0],
                    input_value_info.type.tensor_type.elem_type,
                    input_shape
                )
                if output_value_info:
                    g.value_info.remove(output_value_info)
                g.value_info.extend([new_output_value_info])
                processed = True
                continue
            elif auto_pad == 'VALID':
                pads = [0, 0, 0, 0]
            else:
                print("Unrecognized auto_pad value: " + str(auto_pad))
                exit(1)
        kernel_value_info = helper.find_value_by_name(g, node.input[1])
        _, kernel_shape = helper.find_size_shape_from_value(kernel_value_info)
        if not input_shape or not kernel_shape:
            continue
        strides = helper.get_attribute_by_name(node, 'strides').ints
        if not pads:
            pads = helper.get_attribute_by_name(node, 'pads').ints
        dilation = helper.get_attribute_by_name(node, 'dilations').ints

        # Pytorch model has the case where strides only have one number
        if len(strides) == 1:
            return strides.append(strides[0])
        if len(dilation) == 1:
            return dilation.append(dilation[0])

        H = math.floor((input_shape[2]+pads[0]+pads[2]-\
            dilation[0]*(kernel_shape[2]-1)-1)/strides[0]+1)
        W = math.floor((input_shape[3]+pads[1]+pads[3]-\
            dilation[1]*(kernel_shape[3]-1)-1)/strides[1]+1)
        output_shape = [input_shape[0], kernel_shape[0], H, W]

        new_output_value_info = onnx.helper.make_tensor_value_info(
            node.output[0],
            input_value_info.type.tensor_type.elem_type,
            output_shape
        )

        processed = True

        if output_value_info:
            g.value_info.remove(output_value_info)
        g.value_info.extend([new_output_value_info])

    return processed


def inference_split_shape(g):
    processed = False
    for node in g.node:
        if node.op_type != 'Split':
            continue
        
        input_val_info = helper.find_value_by_name(g, node.input[0])
        if not input_val_info:
            input_val_info = helper.find_input_by_name(g, node.input[0])
        if not input_val_info:
            continue

        _, input_shape = helper.find_size_shape_from_value(input_val_info)
        if not input_shape:
            continue

        output_val_names = list(node.output)
        output_vals = [helper.find_value_by_name(g, val_name) for val_name in output_val_names]

        output_shapes = [helper.find_size_shape_from_value(output_val)[1] for output_val in output_vals]
        if not any([len(s) == 0 for s in output_shapes]):
            continue

        for att in node.attribute:
            if att.name == 'axis':
                axis = att.i
            else:
                split = list(att.ints)
        
        new_output_vals = []
        for i in range(len(output_val_names)):
            new_shape = list(input_shape)
            new_shape[axis] = split[i]
            new_output_val = onnx.helper.make_tensor_value_info(
                output_val_names[i],
                input_val_info.type.tensor_type.elem_type,
                new_shape
            )
            new_output_vals.append(new_output_val)
        
        for val in output_vals:
            if val is not None:
                g.value_info.remove(val)
        g.value_info.extend(new_output_vals)

        processed = True
    
    return processed


def parse_shape_change_input(s: str):
    """The input should be like 'input 1 1 224 224'.
    """
    s_list = s.split(' ')
    if len(s_list) < 2:
        print("Cannot parse the shape change input: {}".format(s))
        return None
    shape = []
    for i in range(1, len(s_list)):
        shape.append(int(s_list[i]))
    return s_list[0], shape

def change_input_shape(g, target_list):
    for target in target_list:
        try:
            name, shape = parse_shape_change_input(target)
            input_value = helper.find_input_by_name(g, name)
            if input_value is None:
                print("Cannot find input {}".format(name))
                continue
            if len(shape) != len(input_value.type.tensor_type.shape.dim):
                print("The dimension doesn't match for input {}".format(name))
                continue
            for i in range(len(shape)):
                input_value.type.tensor_type.shape.dim[i].dim_value = shape[i]
        except TypeError:
            # This happens when the parser function returns None.
            continue
        except ValueError:
            # This happens when the input cannot be converter into int
            print("Cannot parse {} into name and int".format(target))
            continue

def change_output_shape(g, target_list):
    for target in target_list:
        try:
            name, shape = parse_shape_change_input(target)
            output_value = helper.find_output_by_name(g, name)
            if output_value is None:
                print("Cannot find output {}".format(name))
                continue
            if len(shape) != len(output_value.type.tensor_type.shape.dim):
                print("The dimension doesn't match for output {}".format(name))
                continue
            for i in range(len(shape)):
                output_value.type.tensor_type.shape.dim[i].dim_value = shape[i]
        except TypeError:
            # This happens when the parser function returns None.
            continue
        except ValueError:
            # This happens when the input cannot be converter into int
            print("Cannot parse {} into name and int".format(target))
            continue

def add_nop_conv_after(g, value_names):
    """Add do-nothing depthwise Conv nodes after the given value info. It will\\
    take the given names as the inputs of the new node and replace the inputs\\
    of the following nodes.

    :param g: the graph\\
    :param value_names: a list of string which are the names of value_info.
    """
    for value_name in value_names:
        # Find the value first
        value = helper.find_value_by_name(g, value_name)
        if value is None:
            value = helper.find_input_by_name(g, value_name)
        if value is None:
            value = helper.find_output_by_name(g, value_name)
        if value is None:
            print("Cannot find an value_info named {}".format(value_name))
            continue
        # Get the channel number from value info
        shape = helper.get_shape_from_value_info(value)
        channel = shape[1]
        # Construct 4 weights
        node_name = value_name + "_nop_conv"
        ones = [1.0] * channel
        weight_node = helper.list_to_constant(node_name + "_weight", [channel, 1, 1, 1], ones)
        # Construct BN node
        conv_node = onnx.helper.make_node(
            "Conv",
            [value_name,
            weight_node.output[0]],
            [node_name],
            name = node_name,
            dilations = [1, 1],
            group = channel,
            kernel_shape = [1, 1],
            pads = [0, 0, 0, 0],
            strides = [1, 1]
        )
        # Reconnect the graph
        following_nodes = helper.find_following_nodes_by_input_value_name(g, value_name)
        if len(following_nodes) > 0:
            for following_node in following_nodes:
                replace_node_input(following_node, value_name, node_name)
        else:
            # If the node is the output, replace the output with the previous input.
            new_value = onnx.helper.make_tensor_value_info(
                node_name,
                value.type.tensor_type.elem_type,
                shape
            )
            output_values = []
            while len(g.output):
                output_values.append(g.output.pop())
            while output_values:
                output_value = output_values.pop()
                if output_value.name == value_name:
                    g.output.extend([new_value])
                else:
                    g.output.extend([output_value])
        # Add node to the graph
        g.node.extend([conv_node, weight_node])
    topological_sort(g)

def add_shift_conv_after(g, target_node_name, shift_value):
    """Add do-nothing depthwise Conv nodes after the given value info. It will\\
    take the given names as the inputs of the new node and replace the inputs\\
    of the following nodes.

    :param g: the graph\\
    :param target_node_name: a string which are the names of value_info.
    """

    # Find the value first
    value = helper.find_value_by_name(g, target_node_name)
    if value is None:
        value = helper.find_input_by_name(g, target_node_name)
    if value is None:
        value = helper.find_output_by_name(g, target_node_name)
    if value is None:
        print("Cannot find an value_info named {}".format(target_node_name))
        return
    # Get the channel number from value info
    shape = helper.get_shape_from_value_info(value)
    channel = shape[1]
    # Construct 4 weights
    node_name = target_node_name + "_nop_conv"
    ones = [1.0] * channel
    shifts = [shift_value] * channel
    weight_node = helper.list_to_constant(node_name + "_weight", [channel, 1, 1, 1], ones)
    bias_node = helper.list_to_constant(node_name + "_bias", [channel], shifts)
    # Construct BN node
    conv_node = onnx.helper.make_node(
        "Conv",
        [target_node_name,
        weight_node.output[0],
        bias_node.output[0]],
        [node_name],
        name = node_name,
        dilations = [1, 1],
        group = channel,
        kernel_shape = [1, 1],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    )
    # Reconnect the graph
    following_nodes = helper.find_following_nodes_by_input_value_name(g, target_node_name)
    if len(following_nodes) > 0:
        for following_node in following_nodes:
            replace_node_input(following_node, target_node_name, node_name)
    else:
        # If the node is the output, replace the output with the previous input.
        new_value = onnx.helper.make_tensor_value_info(
            node_name,
            value.type.tensor_type.elem_type,
            shape
        )
        output_values = []
        while len(g.output):
            output_values.append(g.output.pop())
        while output_values:
            output_value = output_values.pop()
            if output_value.name == target_node_name:
                g.output.extend([new_value])
            else:
                g.output.extend([output_value])
    # Add node to the graph
    g.node.extend([conv_node, weight_node, bias_node])

    topological_sort(g)

def add_nop_bn_after(g, value_names):
    """Add do-nothing BatchNormalization nodes after the given value info. It will\\
    take the given names as the inputs of the new node and replace the inputs\\
    of the following nodes.

    :param g: the graph\\
    :param value_names: a list of string which are the names of value_info.
    """
    for value_name in value_names:
        # Find the value first
        value = helper.find_value_by_name(g, value_name)
        if value is None:
            value = helper.find_input_by_name(g, value_name)
        if value is None:
            value = helper.find_output_by_name(g, value_name)
        if value is None:
            print("Cannot find an value_info named {}".format(value_name))
            continue
        # Get the channel number from value info
        shape = helper.get_shape_from_value_info(value)
        channel = shape[1]
        # Construct 4 weights
        node_name = value_name + "_nop_bn"
        ones = [1.0] * channel
        zeros = [0.0] * channel
        scale_node = helper.list_to_constant(node_name + "_scale", [channel], ones)
        bias_node = helper.list_to_constant(node_name + "_bias", [channel], zeros)
        mean_node = helper.list_to_constant(node_name + "_mean", [channel], zeros)
        var_node = helper.list_to_constant(node_name + "_var", [channel], ones)
        # Construct BN node
        bn_node = onnx.helper.make_node(
            "BatchNormalization",
            [value_name,
            scale_node.output[0],
            bias_node.output[0],
            mean_node.output[0],
            var_node.output[0]],
            [node_name],
            name = node_name
        )
        # Reconnect the graph
        following_nodes = helper.find_following_nodes_by_input_value_name(g, value_name)
        if len(following_nodes) > 0:
            for following_node in following_nodes:
                replace_node_input(following_node, value_name, node_name)
        else:
            # If the node is the output, replace the output with the previous input.
            new_value = onnx.helper.make_tensor_value_info(
                node_name,
                value.type.tensor_type.elem_type,
                shape
            )
            output_values = []
            while len(g.output):
                output_values.append(g.output.pop())
            while output_values:
                output_value = output_values.pop()
                if output_value.name == value_name:
                    g.output.extend([new_value])
                else:
                    g.output.extend([output_value])
        # Add node to the graph
        g.node.extend([bn_node, scale_node, bias_node, mean_node, var_node])
    topological_sort(g)

def duplicate_shared_Flatten(g):
    """To feed our compiler, bind Flatten with Gemm. If the output of one\\
    Flatten goes to two Gemm nodes, duplicate the Flatten.

    :param g: the graph
    """
    for node in g.node:
        # Find a Flatten node
        if node.op_type != 'Flatten':
            continue
        # Check Flatten outputs. Get following Gemm
        output_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        if len(output_nodes) < 2:
            continue
        gemm_nodes = []
        for output_node in output_nodes:
            if output_node.op_type == 'Gemm':
                gemm_nodes.append(output_node)
        if len(gemm_nodes) < 2:
            continue
        # Process all the Gemm nodes except for the first one.
        for i in range(1, len(gemm_nodes)):
            # Duplicate
            new_flatten_name = node.name + "_copy" + str(i)
            new_flatten_node = onnx.helper.make_node(
                "Flatten",
                node.input,
                [new_flatten_name],
                name=new_flatten_name,
                axis=1
            )
            # Connect new graph
            replace_node_input(gemm_nodes[i], node.output[0], new_flatten_name)
            g.node.extend([new_flatten_node])
    topological_sort(g)

def deconv_to_conv_info_extraction(input_size, node_proto):
    """Extract the information needed for deconv split.

    :param input_size: input shape of the deconv node.\\
    :param node_proto: the deconv node proto.\\
    :return: a dictionary of extracted params.
    """
    attr = dict()
    # Get attributes from Deconv node
    attr["auto_pad"] = helper.get_var_attribute_by_name(node_proto, "auto_pad", "string")
    attr["dilations"] = helper.get_list_attribute_by_name(node_proto, "dilations", "int")
    attr["group"] = helper.get_var_attribute_by_name(node_proto, "group", "int")
    attr["kernel_shape"] = helper.get_list_attribute_by_name(node_proto, "kernel_shape", "int")
    attr["output_padding"] = helper.get_list_attribute_by_name(node_proto, "output_padding", "int")
    attr["pads"] = helper.get_list_attribute_by_name(node_proto, "pads", "int")
    attr["strides"] = helper.get_list_attribute_by_name(node_proto, "strides", "int")
    # Get output_padding
    if attr["output_padding"] is None:
        if attr["auto_pad"] == "SAME_LOWER" or attr["auto_pad"] == "SAME_UPPER":
            attr["output_padding"] = [attr["strides"][0] - 1, attr["strides"][1]]
        else:
            attr["output_padding"] = [max(attr["strides"][0] - attr["kernel_shape"][0], 0),
                                      max(attr["strides"][1] - attr["kernel_shape"][1], 0)]
    # Calculate conv_padding
    if attr["auto_pad"] == "SAME_LOWER" or attr["auto_pad"] == "SAME_UPPER":
        pad1_h = attr["kernel_shape"][0] - (attr["kernel_shape"][0] - 1) // 2 - 1
        pad1_w = attr["kernel_shape"][1] - (attr["kernel_shape"][1] - 1) // 2 - 1
        head_h = min(attr["kernel_shape"][0] // 2, (attr["output_padding"][0] + 1) // 2)
        head_w = min(attr["kernel_shape"][1] // 2, (attr["output_padding"][1] + 1) // 2)
        tail_h = attr["output_padding"][0] - head_h
        tail_w = attr["output_padding"][1] - head_w
        attr["conv_pads"] = [pad1_h + head_h, pad1_w + head_w, pad1_h + tail_h, pad1_w + tail_w]
    elif attr["pads"] is not None:
        sum_of_pads = sum(attr["pads"])
        if sum_of_pads == 0:
            # Valid padding
            pad1_h = attr["kernel_shape"][0] - 0 - 1
            pad1_w = attr["kernel_shape"][1] - 0 - 1
            head_h = 0
            head_w = 0
            tail_h = attr["output_padding"][0] - head_h
            tail_w = attr["output_padding"][1] - head_w
            attr["conv_pads"] = [pad1_h + head_h, pad1_w + head_w, pad1_h + tail_h, pad1_w + tail_w]
        else:
            # Calculate output shape
            tmp_output_shape = [0, 0]
            tmp_output_shape[0] = attr["strides"][0] * (input_size[2] - 1) + attr["output_padding"][0] + attr["kernel_shape"][0] - attr["pads"][0] - attr["pads"][2]
            tmp_output_shape[1] = attr["strides"][1] * (input_size[3] - 1) + attr["output_padding"][1] + attr["kernel_shape"][1] - attr["pads"][1] - attr["pads"][3]
            # Calculate real conv output shape
            tmp_center_shape = [0, 0]
            tmp_center_shape[0] = (input_size[2] - 1) * attr["strides"][0] + 1
            tmp_center_shape[1] = (input_size[3] - 1) * attr["strides"][1] + 1
            # Calculate padding
            total_padding = [0, 0]
            total_padding[0] = tmp_output_shape[0] - tmp_center_shape[0] + attr["kernel_shape"][0] - 1
            total_padding[1] = tmp_output_shape[1] - tmp_center_shape[1] + attr["kernel_shape"][1] - 1
            if total_padding[0] < 0 or total_padding[1] < 0:
                raise RuntimeError(node_proto.name + " cannot infer conv padding.")
            conv_pads_ = [0] * 4
            conv_pads_[0] = total_padding[0] // 2
            conv_pads_[1] = total_padding[1] // 2
            conv_pads_[2] = total_padding[0] - total_padding[0] // 2
            conv_pads_[3] = total_padding[1] - total_padding[1] // 2
            attr["conv_pads"] = conv_pads_
    else:
        pad1_h = attr["kernel_shape"][0] - 0 - 1
        pad1_w = attr["kernel_shape"][1] - 0 - 1
        head_h = 0
        head_w = 0
        tail_h = attr["output_padding"][0] - head_h
        tail_w = attr["output_padding"][1] - head_w
        attr["conv_pads"] = [pad1_h + head_h, pad1_w + head_w, pad1_h + tail_h, pad1_w + tail_w]
    return attr

def split_ConvTranspose(model):
    """To feed our compiler, split ConvTranspose into Upsample and Conv.

    :param model: the model
    """
    node_to_delete = []
    # Change model properties for upsample.
    if model.ir_version < 3:
        print("Warning: Current model IR version is not fully supported.")
    model.ir_version = 4
    model.opset_import[0].version = 9
    g = model.graph
    # Get a Convtranspose layer
    for node in g.node:
        # Find a Flatten node
        if node.op_type != 'ConvTranspose':
            continue
        # Check auto_pad
        auto_pad_proto = helper.get_attribute_by_name(node, "auto_pad")
        if auto_pad_proto is not None:
            print("Currently not split auto_pad ConvTranspose")
            continue
        # Check output_shape
        output_shape_proto = helper.get_attribute_by_name(node, "output_shape")
        if output_shape_proto is not None:
            print("Currently not split output_shape ConvTranspose")
            continue
        # Get input shape
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            input_value = helper.find_input_by_name(g, node.input[0])
        if input_value is None:
            print("Cannot get value info named {}.".format(node.input[0]))
            exit(1)
        input_shape = helper.get_shape_from_value_info(input_value)
        # Get attrbutes
        attr = deconv_to_conv_info_extraction(input_shape, node)
        # Generate Upsample scales
        upsample_output_shape = list(input_shape)
        upsample_output_shape[2] = (input_shape[2] - 1) * attr["strides"][0] + 1
        upsample_output_shape[3] = (input_shape[3] - 1) * attr["strides"][1] + 1
        upsample_node_name = node.name + "_inner_upsample"
        upsample_scale_name = upsample_node_name + "_scales"
        scales_np = np.ones([4]).astype('float32')
        scales_np[2] = float(upsample_output_shape[2]) / input_shape[2]
        scales_np[3] = float(upsample_output_shape[3]) / input_shape[3]
        scales_node = helper.numpy_to_constant(upsample_scale_name, scales_np)
        # Generate a Upsample layer and an internal value info
        upsample_node = onnx.helper.make_node(
            "Upsample",
            [node.input[0], upsample_scale_name],
            [upsample_node_name],
            name=upsample_node_name,
            mode="zeros"
        )
        upsample_value_info = onnx.helper.make_tensor_value_info(
            upsample_node_name,
            input_value.type.tensor_type.elem_type,
            upsample_output_shape
        )
        # Check the weight layer, it may need a transpose
        if attr["group"] != input_shape[1]:
            weight_node = helper.find_node_by_output_name(g, node.input[1])
            weight_np = helper.constant_to_numpy(weight_node)
            new_weight_np = np.transpose(weight_np, [1, 0, 2, 3])
            new_weight_node = helper.numpy_to_constant(node.input[1], new_weight_np)
            node_to_delete.append(weight_node)
            g.node.extend([new_weight_node])
            value = helper.find_value_by_name(g, node.input[1])
            g.value_info.remove(value)
        # Generate a Conv layer
        conv_node_name = node.name + "_inner_conv"
        conv_node_input = [upsample_node_name]
        conv_node_input.extend(node.input[1:])
        conv_node = onnx.helper.make_node(
            "Conv",
            conv_node_input,
            [node.output[0]],
            name=conv_node_name,
            pads=[int(i) for i in attr["conv_pads"]],
            dilations=[int(i) for i in attr["dilations"]],
            group=int(attr["group"]),
            kernel_shape=[int(i) for i in attr["kernel_shape"]],
            strides=[int(1), int(1)]
        )
        # Reconnect the graph
        g.node.extend([scales_node, upsample_node, conv_node])
        g.value_info.extend([upsample_value_info])
        node_to_delete.append(node)
    # Delete useless nodes
    for node in node_to_delete:
        g.node.remove(node)
    topological_sort(g)

def add_bn_on_skip_branch(g):
    for n in g.node:
        # Find merge node (Add)
        if n.op_type != 'Add':
            continue
        if len(n.input) != 2:
            continue
        # TODO: Still need to consider more cases
        # Check if skip branch exist
        input_node_a = helper.find_node_by_output_name(g, n.input[0])
        output_of_input_node_a = helper.find_nodes_by_input_name(g, input_node_a.output[0])
        input_node_b = helper.find_node_by_output_name(g, n.input[1])
        output_of_input_node_b = helper.find_nodes_by_input_name(g, input_node_b.output[0])
        if len(output_of_input_node_a) == 1 and len(output_of_input_node_b) == 1:
            continue
        if len(output_of_input_node_a) == 2:
            split_node = input_node_a
        elif len(output_of_input_node_b) == 2:
            split_node = input_node_b
        else:
            continue
        # Get the channel number from value info
        value_name = split_node.output[0]
        value = helper.find_value_by_name(g, value_name)
        shape = helper.get_shape_from_value_info(value)
        channel = shape[1]
        # Construct 4 weights
        node_name = value_name + "_nop_bn"
        ones = [1.0] * channel
        zeros = [0.0] * channel
        scale_node = helper.list_to_constant(node_name + "_scale", [channel], ones)
        bias_node = helper.list_to_constant(node_name + "_bias", [channel], zeros)
        mean_node = helper.list_to_constant(node_name + "_mean", [channel], zeros)
        var_node = helper.list_to_constant(node_name + "_var", [channel], ones)
        # Construct BN node
        bn_node = onnx.helper.make_node(
            "BatchNormalization",
            [value_name,
            scale_node.output[0],
            bias_node.output[0],
            mean_node.output[0],
            var_node.output[0]],
            [node_name],
            name = node_name
        )
        # Reconnect the graph
        replace_node_input(n, value_name, node_name)
        # Add node to the graph
        g.node.extend([bn_node, scale_node, bias_node, mean_node, var_node])
    topological_sort(g)

def add_bn_before_add(g):
    for n in g.node:
        # Find merge node (Add)
        if n.op_type != 'Add':
            continue
        if len(n.input) != 2:
            continue
        # Get two inputs
        input_node_a = helper.find_node_by_output_name(g, n.input[0])
        input_node_b = helper.find_node_by_output_name(g, n.input[1])
        def add_bn_after(prev_node):
            # Get the channel number from value info
            value_name = prev_node.output[0]
            value = helper.find_value_by_name(g, value_name)
            shape = helper.get_shape_from_value_info(value)
            channel = shape[1]
            # Construct 4 weights
            node_name = value_name + "_nop_bn"
            ones = [1.0] * channel
            zeros = [0.0] * channel
            scale_node = helper.list_to_constant(node_name + "_scale", [channel], ones)
            bias_node = helper.list_to_constant(node_name + "_bias", [channel], zeros)
            mean_node = helper.list_to_constant(node_name + "_mean", [channel], zeros)
            var_node = helper.list_to_constant(node_name + "_var", [channel], ones)
            # Construct BN node
            bn_node = onnx.helper.make_node(
                "BatchNormalization",
                [value_name,
                scale_node.output[0],
                bias_node.output[0],
                mean_node.output[0],
                var_node.output[0]],
                [node_name],
                name = node_name
            )
            # Reconnect the graph
            replace_node_input(n, value_name, node_name)
            # Add node to the graph
            g.node.extend([bn_node, scale_node, bias_node, mean_node, var_node])
        if not (input_node_a.op_type == 'BatchNormalization' and len(helper.find_following_nodes_by_input_value_name(g, input_node_a.output[0])) == 1):
            add_bn_after(input_node_a)
        if not (input_node_b.op_type == 'BatchNormalization' and len(helper.find_following_nodes_by_input_value_name(g, input_node_b.output[0])) == 1):
            add_bn_after(input_node_b)
    topological_sort(g)

def add_bn_before_activation(g):
    activation_nodes = set(['Relu', 'Clip', 'PRelu', 'LeakyRelu'])
    previous_nodes = set(['Conv', 'BatchNormalization'])
    for n in g.node:
        # Find activation node
        if n.op_type not in activation_nodes:
            continue
        # Get input
        input_node = helper.find_node_by_output_name(g, n.input[0])
        if input_node is None or input_node.op_type in previous_nodes:
            continue
        def add_bn_after(prev_node):
            # Get the channel number from value info
            value_name = prev_node.output[0]
            value = helper.find_value_by_name(g, value_name)
            shape = helper.get_shape_from_value_info(value)
            channel = shape[1]
            # Construct 4 weights
            node_name = value_name + "_nop_bn"
            ones = [1.0] * channel
            zeros = [0.0] * channel
            scale_node = helper.list_to_constant(node_name + "_scale", [channel], ones)
            bias_node = helper.list_to_constant(node_name + "_bias", [channel], zeros)
            mean_node = helper.list_to_constant(node_name + "_mean", [channel], zeros)
            var_node = helper.list_to_constant(node_name + "_var", [channel], ones)
            # Construct BN node
            bn_node = onnx.helper.make_node(
                "BatchNormalization",
                [value_name,
                scale_node.output[0],
                bias_node.output[0],
                mean_node.output[0],
                var_node.output[0]],
                [node_name],
                name = node_name
            )
            # Reconnect the graph
            replace_node_input(n, value_name, node_name)
            # Add node to the graph
            g.node.extend([bn_node, scale_node, bias_node, mean_node, var_node])
        add_bn_after(input_node)
    topological_sort(g)

def pytorch_check_initializer_as_input(g):
    if len(g.input) < len(g.initializer):
        raise RuntimeError("You need to add option `keep_initializers_as_inputs=True` while exporting the model!")

def rename_output_name(g, original_name, new_name):
    # Output
    output_value = helper.find_output_by_name(g, original_name)
    if output_value is None:
        logging.error("Cannot find output value named " + original_name)
        return
    output_value.name = new_name
    # Value Info
    value_info = helper.find_value_by_name(g, original_name)
    if value_info is not None:
        value_info.name = new_name
    # Node output
    node = helper.find_node_by_output_name(g, original_name)
    node.output[0] = new_name
    # Node input
    nodes = helper.find_nodes_by_input_name(g, original_name)
    for node in nodes:
        replace_node_input(node, original_name, new_name)
