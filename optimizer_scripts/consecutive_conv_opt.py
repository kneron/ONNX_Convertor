import numpy as np
import onnx
import sys

from tools.other import topological_sort
from tools import helper

def fuse_bias_in_consecutive_1x1_conv(g):
    for second in g.node:
        # Find two conv
        if second.op_type != 'Conv':
            continue
        first = helper.find_node_by_output_name(g, second.input[0])
        if first is None or first.op_type != 'Conv':
            continue
        # Check if the first one has only one folloing node
        if len(helper.find_following_nodes_by_input_value_name(g, first.output[0])) != 1:
            continue
        # If first node has no bias, continue
        if len(first.input) == 2:
            continue
        # Check their kernel size
        first_kernel_shape = helper.get_list_attribute_by_name(first, 'kernel_shape', 'int')
        second_kernel_shape = helper.get_list_attribute_by_name(second, 'kernel_shape', 'int')
        prod = first_kernel_shape[0] * first_kernel_shape[1] * second_kernel_shape[0] * second_kernel_shape[1]
        if prod != 1:
            continue
        print('Found: ', first.name, ' ', second.name)
        # Get bias of the nodes
        first_bias_node = helper.find_node_by_output_name(g, first.input[2])
        second_weight_node = helper.find_node_by_output_name(g, second.input[1])
        second_bias_node = helper.find_node_by_output_name(g, second.input[2])
        first_bias = helper.constant_to_numpy(first_bias_node)
        second_weight = helper.constant_to_numpy(second_weight_node)
        second_bias = helper.constant_to_numpy(second_bias_node)
        # Calculate the weight for second node
        first_bias = np.reshape(first_bias, (1, first_bias.size))
        second_weight = np.reshape(second_weight, (second_weight.shape[0], second_weight.shape[1]))
        second_weight = np.transpose(second_weight)
        new_second_bias = second_bias + np.matmul(first_bias, second_weight)
        new_second_bias = np.reshape(new_second_bias, (new_second_bias.size,))
        # Generate new weight
        new_first_bias = np.reshape(first_bias, (first_bias.size, ))
        for i in range(new_first_bias.shape[0]):
            new_first_bias[i] = 0.0
        new_first_bias_node = helper.numpy_to_constant(first_bias_node.output[0], new_first_bias)
        new_second_bias_node = helper.numpy_to_constant(second_bias_node.output[0], new_second_bias)
        # Delete old weight and add new weights
        g.node.remove(first_bias_node)
        g.node.remove(second_bias_node)
        g.node.extend([new_first_bias_node, new_second_bias_node])
    topological_sort(g)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(1)
    m = onnx.load(sys.argv[1])
    fuse_bias_in_consecutive_1x1_conv(m.graph)
    onnx.save(m, sys.argv[2])