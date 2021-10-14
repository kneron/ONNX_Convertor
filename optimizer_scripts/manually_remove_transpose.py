from tools import other, helper, modhelper
from onnx import optimizer
import sys
import onnx
import onnx.helper


# Reorder Conv-Reshape- Add and fuse Conv Add
def reorder_Conv_Reshape_Add(g):
    for add_node in g.node:
        # Find an Add
        if add_node.op_type != 'Add':
            continue
        # Check for Reshape
        reshape_node = helper.find_node_by_output_name(g, add_node.input[0])
        if reshape_node is None:
            continue
        if reshape_node.op_type != 'Reshape':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, add_node.input[0])) > 1:
            continue
        # Check for Conv
        conv_node = helper.find_node_by_output_name(g, reshape_node.input[0])
        if conv_node is None:
            continue
        if conv_node.op_type != 'Conv':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, reshape_node.input[0])) > 1:
            continue
        if len(conv_node.input) != 2:
            continue
        # Generate bias
        add_weight_node = helper.find_node_by_output_name(g, add_node.input[1])
        if add_weight_node is None:
            continue
        if add_weight_node.op_type != 'Constant':
            continue
        _, add_weight_list = helper.constant_to_list(add_weight_node)
        new_weight_node = helper.list_to_constant(conv_node.name + '_bias', (len(add_weight_list), ), add_weight_list)
        # Add bias to Conv
        conv_node.input.append(new_weight_node.output[0])
        g.node.append(new_weight_node)
        # Reconnect Tranpose to Add's children
        for child_node in helper.find_following_nodes_by_input_value_name(g, add_node.output[0]):
            modhelper.replace_node_input(child_node, add_node.output[0], reshape_node.output[0])
        # Remove Add
        g.node.remove(add_node)

# Reshape going down
def swap_Reshape_and_LeakyRelu(g):
    for leaky_node in g.node:
        # Find leaky relu reshape pattern
        if leaky_node.op_type != "LeakyRelu":
            continue
        reshape_node = helper.find_node_by_output_name(g, leaky_node.input[0])
        if reshape_node is None:
            continue
        if reshape_node.op_type != 'Reshape':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, leaky_node.input[0])) > 1:
            continue
        #Swap their position
        for following_node in helper.find_following_nodes_by_input_value_name(g, leaky_node.output[0]):
            modhelper.replace_node_input(following_node, leaky_node.output[0], reshape_node.output[0])
        old_input_of_reshape = reshape_node.input[0]
        reshape_node.input[0] = leaky_node.output[0]
        leaky_node.input[0] = old_input_of_reshape


# Reshape Transpose Reshape Transpose pattern elimination
def eliminate_reshape_transpose_pattern(g):
    for transpose_node in g.node:
        # Find Reshape Transpose pattern
        if transpose_node.op_type != 'Transpose':
            continue
        reshape_node = helper.find_node_by_output_name(g, transpose_node.input[0])
        if reshape_node is None:
            continue
        if reshape_node.op_type != 'Reshape':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, transpose_node.input[0])) > 1:
            continue
        # Reconnect
        for child_node in helper.find_following_nodes_by_input_value_name(g, transpose_node.output[0]):
            modhelper.replace_node_input(child_node, transpose_node.output[0], reshape_node.input[0])
        # Delete both nodes
        g.node.remove(reshape_node)
        g.node.remove(transpose_node)

def eliminate_Transpose_surround_Concat(g):
    # Find concat
    for concat_node in g.node:
        if concat_node.op_type != 'Concat':
            continue
        # Find all the Transpose before and after
        input_nodes = [helper.find_node_by_output_name(g, input_name) for input_name in concat_node.input]
        failed = False
        for input_node in input_nodes:
            if input_node.op_type != "Transpose":
                failed = True
                break
        if failed:
            continue
        output_nodes = helper.find_following_nodes_by_input_value_name(g, concat_node.output[0])
        for output_node in output_nodes:
            if output_node.op_type != "Transpose":
                failed = True
                break
        if failed:
            continue
        # Change the Concat axis
        axis_attr = helper.get_attribute_by_name(concat_node, 'axis')
        axis_attr.i = 1
        # Reconnect the graph
        while len(concat_node.input) > 0:
            concat_node.input.pop()
        for input_node in input_nodes:
            concat_node.input.append(input_node.input[0])
        for output_node in output_nodes:
            for following_node in helper.find_following_nodes_by_input_value_name(g, output_node.output[0]):
                modhelper.replace_node_input(following_node, output_node.output[0], concat_node.output[0])
        # Delete the tranpose nodes
        for n in input_nodes:
            g.node.remove(n)
        for n in output_nodes:
            g.node.remove(n)

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



m = onnx.load(sys.argv[1])
reorder_Conv_Reshape_Add(m.graph)
swap_Reshape_and_LeakyRelu(m.graph)
eliminate_reshape_transpose_pattern(m.graph)

add_pair_of_transpose_nodes(m.graph)
eliminate_Transpose_surround_Concat(m.graph)

other.topological_sort(m.graph)
while len(m.graph.value_info) != 0:
    m.graph.value_info.pop()
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, sys.argv[2])