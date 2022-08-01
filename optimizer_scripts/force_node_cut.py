import onnx
import onnx.utils
from onnx import optimizer
import argparse
import sys, os
import logging

from tools import helper
from tools import other
from tools import eliminating
from tools import replacing

if __name__ != '__main__':
    logging.error("This module can only be called directly")
    exit(1)

def force_node_remove(g, values_to_remove, shapes):
    node_to_remove = []
    # Find the value info
    for i in range(len(values_to_remove)):
        value_name = values_to_remove[i]
        value = helper.find_output_by_name(g, value_name)
        if value is None:
            value = helper.find_value_by_name(g, value_name)
            if value is None:
                value = onnx.helper.make_tensor_value_info(value_name, 0, shapes[i])
            g.output.append(value)
        # Remove the nodes following
        for node in g.node:
            # Find the node to delete
            if value_name in node.input:
                node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)

def parse_shapes(shapes_str):
    shapes = []
    for shape_str in shapes_str:
        s_list = shape_str.split(' ')
        if len(s_list) < 1:
            print("Cannot parse the shape change input: {}".format(shape_str))
            return None
        shape = []
        for i in range(0, len(s_list)):
            shape.append(int(s_list[i]))
        shapes.append(shape)
    return shapes


parser = argparse.ArgumentParser(description="Edit an ONNX model.")
parser.add_argument('in_file', type=str, help='input ONNX FILE')
parser.add_argument('out_file', type=str, help="ouput ONNX FILE")
parser.add_argument('-c', '--cut', dest='cut_values',type=str, nargs='+', required=True, help="remove nodes since the given value_info(inclusive)")
parser.add_argument('-s', '--shapes', type=str, nargs='+', help="The shape of the given value_info(inclusive)")

args = parser.parse_args()

# Load model
m = onnx.load(args.in_file)
# Format input/output
other.format_input_output(m.graph)
eliminating.eliminate_empty_value_infos(m.graph)
replacing.replace_initializer_with_Constant(m.graph, False)
# Prepare shapes
eliminating.clear_value_infos(m.graph)
m = onnx.shape_inference.infer_shapes(m)
eliminating.eliminate_empty_value_infos(m.graph)
# Remove the nodes
if args.shapes:
    shapes = parse_shapes(args.shapes)
else:
    shapes = None
force_node_remove(m.graph, args.cut_values, shapes)
eliminating.eliminate_not_connected_nodes(m.graph)
eliminating.eliminate_not_connected_outputs(m.graph)
other.topological_sort(m.graph)
eliminating.clear_value_infos(m.graph)
m = onnx.utils.polish_model(m)

# Save model
onnx.save(m, args.out_file)
