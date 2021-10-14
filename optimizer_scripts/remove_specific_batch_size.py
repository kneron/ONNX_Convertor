from collections import deque
import logging

from tools import other, replacing, helper, modhelper
from onnx import optimizer
import sys
import onnx

def remove_reshape_of_batch_change(g):
    for n in g.node:
        # Check the operator type
        if n.op_type != 'Reshape':
            continue
        # Get the input and output shape
        input_value = helper.find_value_by_name(g, n.input[0])
        input_shape = helper.get_shape_from_value_info(input_value)
        output_value = helper.find_value_by_name(g, n.output[0])
        output_shape = helper.get_shape_from_value_info(output_value)
        # Check the shape change
        if len(input_shape) == len(output_shape) - 1:
            # 25 x a x b -> 1 x 25 x a x b
            if output_shape[0] != 1:
                continue
            not_matched = False
            for i in range(len(input_shape)):
                if input_shape[i] != output_shape[i + 1]:
                    not_matched = True
                    break
            if not_matched:
                continue
        elif len(input_shape) == len(output_shape) + 1:
            # 1 x 25 x a x b -> 25 x a x b
            if input_shape[0] != 1:
                continue
            not_matched = False
            for i in range(len(output_shape)):
                if input_shape[i + 1] != output_shape[i]:
                    not_matched = True
                    break
            if not_matched:
                continue
        else:
            continue
        # Reconnect the graph
        following_nodes = helper.find_following_nodes_by_input_value_name(g, n.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, n.output[0], n.input[0])
        # Delete the value_info
        g.value_info.remove(output_value)
        # Delete the node
        g.node.remove(n)

def concat_axis_adjust(g):
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
    # Fix the final reshape
    the_reshape_shape = helper.find_node_by_node_name(g, 'const_fold_opt__977_kn')
    the_reshape_shape_np = helper.constant_to_numpy(the_reshape_shape)
    the_reshape_shape_np = the_reshape_shape_np[1:]
    new_reshape_shape = helper.numpy_to_constant('const_fold_opt__977_kn', the_reshape_shape_np)
    g.node.remove(the_reshape_shape)
    g.node.extend([new_reshape_shape])
    # Fix the output
    v = onnx.helper.make_tensor_value_info(g.output[0].name, g.output[0].type.tensor_type.elem_type, (25, 96))
    g.output.pop()
    g.output.append(v)


m = onnx.load(sys.argv[1])
remove_reshape_of_batch_change(m.graph)
special_adjust_for_reshape(m.graph)
concat_axis_adjust(m.graph)
other.topological_sort(m.graph)
while len(m.graph.value_info) != 0:
    m.graph.value_info.pop()
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, sys.argv[2])