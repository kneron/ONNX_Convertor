from collections import deque
import logging
import argparse
from onnx import optimizer

from tools import other, helper, eliminating, modhelper
import sys
import onnx

if __name__ != '__main__':
    logging.error("This module can only be called directly")
    exit(1)

def concat_axis_adjust(g):
    input_shape = helper.get_shape_from_value_info(g.input[0])
    if input_shape[0] == 1:
        return
    # Only happens when batch is not 1
    for n in g.node:
        # Check the operator type
        if n.op_type != 'Concat':
            continue
        # Check input shape and axis
        input_value = helper.find_value_by_name(g, n.input[-1])
        input_shape = helper.get_shape_from_value_info(input_value)
        if len(input_shape) == helper.get_var_attribute_by_name(n, 'axis', 'int'):
            helper.get_attribute_by_name(n, 'axis').i = len(input_shape) - 1

def special_adjust_for_reshape(g):
    # Find the Reshape to adjust
    for node in g.node:
        if node.op_type != 'Concat':
            continue
        # Check concat's input
        if len(node.input) < 3:
            continue
        # Get common shape
        shape_count = {}
        for input_name in node.input:
            input_value = helper.find_value_by_name(g, input_name)
            if input_value is None:
                continue
            input_shape = helper.get_shape_from_value_info(input_value)
            if len(input_shape) not in shape_count:
                shape_count[len(input_shape)] = 1
            else:
                shape_count[len(input_shape)] += 1
        if len(shape_count) == 1:
            continue
        # Check invalid shape
        invalid_shape_size = None
        for input_shape_size in shape_count:
            if shape_count[input_shape_size] == 1:
                invalid_shape_size = input_shape_size
                break
        if invalid_shape_size is None:
            continue
        invalid_input_value = None
        for input_name in node.input:
            input_value = helper.find_value_by_name(g, input_name)
            if input_value is None:
                continue
            input_shape = helper.get_shape_from_value_info(input_value)
            if len(input_shape) == invalid_shape_size:
                invalid_input_value = input_value
                break
        # Check the input op_type
        invalid_input_name = invalid_input_value.name
        input_node = helper.find_node_by_output_name(g, invalid_input_name)
        if input_node is None or input_node.op_type != 'Reshape':
            continue
        # Fix the final reshape
        the_reshape_shape = helper.find_node_by_output_name(g, input_node.input[1])
        the_reshape_shape_np = helper.constant_to_numpy(the_reshape_shape)
        the_reshape_shape_np = the_reshape_shape_np[1:]
        new_reshape_shape = helper.numpy_to_constant(input_node.input[1], the_reshape_shape_np)
        g.node.remove(the_reshape_shape)
        g.node.extend([new_reshape_shape])

def fix_the_output_shape(g):
    # Check input batch
    input_shape = helper.get_shape_from_value_info(g.input[0])
    output_shape = helper.get_shape_from_value_info(g.output[0])
    if input_shape[0] != output_shape[0]:
        return
    # Fix the output
    v = onnx.helper.make_tensor_value_info(g.output[0].name, g.output[0].type.tensor_type.elem_type, output_shape[1:])
    g.output.pop()
    g.output.append(v)

def add_pair_of_transpose_nodes(g):
    # Create transpose nodes
    input_name = "StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/concatenate_44/concat:0_kn"
    transpose_a = onnx.helper.make_node(
        "Transpose",
        [input_name],
        [input_name + '_extra_transpose_a'],
        name=input_name + '_extra_transpose_a',
        perm=[0, 3, 1, 2]
        )
    transpose_b = onnx.helper.make_node(
        "Transpose",
        [input_name + '_extra_transpose_a'],
        [input_name + '_extra_transpose_b'],
        name=input_name + '_extra_transpose_b',
        perm=[0, 2, 3, 1]
        )
    # Reconnect the graph
    following_node = helper.find_node_by_output_name(g, "StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/add_58/add:0_kn")
    modhelper.replace_node_input(following_node, input_name, input_name + '_extra_transpose_b')
    # Add to the graph
    g.node.extend([transpose_a, transpose_b])
    # Create transpose nodes
    input_name = "StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/concatenate_45/concat:0_kn"
    transpose_a = onnx.helper.make_node(
        "Transpose",
        [input_name],
        [input_name + '_extra_transpose_a'],
        name=input_name + '_extra_transpose_a',
        perm=[0, 3, 1, 2]
        )
    transpose_b = onnx.helper.make_node(
        "Transpose",
        [input_name + '_extra_transpose_a'],
        [input_name + '_extra_transpose_b'],
        name=input_name + '_extra_transpose_b',
        perm=[0, 2, 3, 1]
        )
    # Reconnect the graph
    following_node = helper.find_node_by_output_name(g, "StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/add_58/add:0_kn")
    modhelper.replace_node_input(following_node, input_name, input_name + '_extra_transpose_b')
    # Add to the graph
    g.node.extend([transpose_a, transpose_b])


parser = argparse.ArgumentParser(description="Edit an ONNX model.")
parser.add_argument('in_file', type=str, help='input ONNX FILE')
parser.add_argument('out_file', type=str, help="ouput ONNX FILE")

args = parser.parse_args()


m = onnx.load(args.in_file)
eliminating.remove_reshape_of_batch_change(m.graph)
special_adjust_for_reshape(m.graph)
fix_the_output_shape(m.graph)
concat_axis_adjust(m.graph)
other.topological_sort(m.graph)
while len(m.graph.value_info) != 0:
    m.graph.value_info.pop()
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
other.reorder_Conv_Reshape_Add(m.graph)
other.swap_Reshape_and_LeakyRelu(m.graph)
eliminating.eliminate_reshape_transpose_pattern(m.graph)
add_pair_of_transpose_nodes(m.graph)
eliminating.eliminate_Transpose_surround_Concat(m.graph)
other.topological_sort(m.graph)
while len(m.graph.value_info) != 0:
    m.graph.value_info.pop()
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, args.out_file)