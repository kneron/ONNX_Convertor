from tools import other, helper, replacing, eliminating
from onnx import optimizer
import sys
import onnx
import onnx.helper
import argparse

# StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/time_distributed_564/Reshape_1_kn
def change_reshape_size(g, rehape_name):
    reshape_node = helper.find_node_by_node_name(g, rehape_name)
    reshape_output = helper.find_value_by_name(g, reshape_node.output[0])
    flatten_node = onnx.helper.make_node(
        "Flatten",
        [reshape_node.input[0]],
        [reshape_node.output[0]],
        name=reshape_node.name,
        axis=1
        )
    g.node.remove(reshape_node)
    g.node.append(flatten_node)
    if reshape_output is not None:
        g.value_info.remove(reshape_output)

parser = argparse.ArgumentParser(description="Edit an ONNX model.")
parser.add_argument('in_file', type=str, help='input ONNX FILE')
parser.add_argument('out_file', type=str, help="ouput ONNX FILE")
parser.add_argument('-i', '--input', dest='input_change', type=str, nargs='+', help="change input shape (e.g. -i 'input_0 1 3 224 224')")
parser.add_argument('-o', '--output', dest='output_change', type=str, nargs='+', help="change output shape (e.g. -o 'input_0 1 3 224 224')")
parser.add_argument('--replace-reshape-with-flatten', dest='replace_reshape', type=str, nargs='+', help="Replace the strange reshape with flatten.")

args = parser.parse_args()

m = onnx.load(args.in_file)
replacing.replace_initializer_with_Constant(m.graph)
m = onnx.utils.polish_model(m)
g = m.graph
other.topological_sort(g)
# Change input and output shapes as requested
if args.input_change is not None:
    other.change_input_shape(g, args.input_change)
if args.output_change is not None:
    other.change_output_shape(g, args.output_change)
if args.replace_reshape is not None:
    for replace_reshape_name in args.replace_reshape:
        change_reshape_size(m.graph, replace_reshape_name)
other.topological_sort(m.graph)
# Reinference the shapes
eliminating.clear_value_infos(m.graph)
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, args.out_file)