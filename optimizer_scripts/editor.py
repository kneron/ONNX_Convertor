import onnx
import onnx.utils
from onnx import optimizer
import argparse

import tools.modhelper as helper
import tools.other as other
import tools.replacing as replacing
# Main process
# Argument parser
parser = argparse.ArgumentParser(description="Edit an ONNX model.\nThe processing sequense is 'delete nodes/values' -> 'add nodes' -> 'change shapes'.\nCutting cannot be done with other operations together")
parser.add_argument('in_file', type=str, help='input ONNX FILE')
parser.add_argument('out_file', type=str, help="ouput ONNX FILE")
parser.add_argument('-c', '--cut', dest='cut_node', type=str, nargs='+', help="remove nodes from the given nodes(inclusive)")
parser.add_argument('--cut-type', dest='cut_type', type=str, nargs='+', help="remove nodes by type from the given nodes(inclusive)")
parser.add_argument('-d', '--delete', dest='delete_node', type=str, nargs='+', help="delete nodes by names and only those nodes")
parser.add_argument('--delete-input', dest='delete_input', type=str, nargs='+', help="delete inputs by names")
parser.add_argument('--delete-output', dest='delete_output', type=str, nargs='+', help="delete outputs by names")
parser.add_argument('-i', '--input', dest='input_change', type=str, nargs='+', help="change input shape (e.g. -i 'input_0 1 3 224 224')")
parser.add_argument('-o', '--output', dest='output_change', type=str, nargs='+', help="change output shape (e.g. -o 'input_0 1 3 224 224')")
parser.add_argument('--add-conv', dest='add_conv', type=str, nargs='+', help='add nop conv using specific input')
parser.add_argument('--add-bn', dest='add_bn', type=str, nargs='+', help='add nop bn using specific input')
parser.add_argument('--rename-output', dest='rename_output', type=str, nargs='+', help='Rename the specific output(e.g. --rename-output old_name new_name)')
parser.add_argument('--pixel-bias-value', dest='pixel_bias_value', type=str, nargs='+', help='(per channel) set pixel value bias bn layer at model front for normalization( e.g. --pixel_bias_value "[104.0, 117.0, 123.0]" )')
parser.add_argument('--pixel-scale-value', dest='pixel_scale_value', type=str, nargs='+', help='(per channel) set pixel value scale bn layer at model front for normalization( e.g. --pixel_scale_value "[0.0078125, 0.0078125, 0.0078125]" )')

args = parser.parse_args()

# Load model and polish
m = onnx.load(args.in_file)
replacing.replace_initializer_with_Constant(m.graph)
m = onnx.utils.polish_model(m)
g = m.graph
other.topological_sort(g)

# Remove nodes according to the given arguments.
if args.delete_node is not None:
    helper.delete_nodes(g, args.delete_node)

if args.delete_input is not None:
    helper.delete_input(g, args.delete_input)

if args.delete_output is not None:
    helper.delete_output(g, args.delete_output)

# Add do-nothing Conv node
if args.add_conv is not None:
    other.add_nop_conv_after(g, args.add_conv)
    other.topological_sort(g)

# Add do-nothing BN node
if args.add_bn is not None:
    other.add_nop_bn_after(g, args.add_bn)
    other.topological_sort(g)

# Add bias scale BN node
if args.pixel_bias_value is not None or args.pixel_scale_value is not None:

    if len(g.input) > 1:
        raise ValueError(" '--pixel-bias-value' and '--pixel-scale-value' only support one input node model currently")
        
    i_n = g.input[0]

    pixel_bias_value = [0] * i_n.type.tensor_type.shape.dim[1].dim_value
    pixel_scale_value = [1] * i_n.type.tensor_type.shape.dim[1].dim_value

    if args.pixel_bias_value is not None and len(args.pixel_bias_value) == 1:
        pixel_bias_value = [float(n) for n in args.pixel_bias_value[0].replace( '[' , '' ).replace( ']' , '' ).split(',')]

    if args.pixel_scale_value is not None and len(args.pixel_scale_value) == 1:
        pixel_scale_value = [float(n) for n in args.pixel_scale_value[0].replace( '[' , '' ).replace( ']' , '' ).split(',')]


    if i_n.type.tensor_type.shape.dim[1].dim_value != len(pixel_bias_value) or  i_n.type.tensor_type.shape.dim[1].dim_value != len(pixel_scale_value):
        raise ValueError("--pixel-bias-value (" + str(pixel_bias_value) + ") and --pixel-scale-value (" + str(pixel_scale_value) + ") should be same as input dimension:" + str(i_n.type.tensor_type.shape.dim[1].dim_value) )
    other.add_bias_scale_bn_after(g, i_n.name,  pixel_bias_value, pixel_scale_value)

# Change input and output shapes as requested
if args.input_change is not None:
    other.change_input_shape(g, args.input_change)
if args.output_change is not None:
    other.change_output_shape(g, args.output_change)

# Cutting nodes according to the given arguments.
if args.cut_node is not None or args.cut_type is not None:
    if args.cut_node is None:
        other.remove_nodes(g, cut_types=args.cut_type)
    elif args.cut_type is None:
        other.remove_nodes(g, cut_nodes=args.cut_node)
    else:
        other.remove_nodes(g, cut_nodes=args.cut_node, cut_types=args.cut_type)
    other.topological_sort(g)

# Rename nodes
if args.rename_output:
    if len(args.rename_output) % 2 != 0:
        helper.logger.warn("Rename output should be paires of names.")
    else:
        for i in range(0, len(args.rename_output), 2):
            other.rename_output_name(g, args.rename_output[i], args.rename_output[i + 1])

# Remove useless nodes
if args.delete_node or args.delete_input or args.input_change or args.output_change:
    # If shape changed during the modification, redo shape inference.
    while(len(g.value_info) > 0):
        g.value_info.pop()
passes = ['extract_constant_to_initializer']
m = optimizer.optimize(m, passes)
g = m.graph
replacing.replace_initializer_with_Constant(g)
other.topological_sort(g)
# Polish and output
m = onnx.utils.polish_model(m)
other.add_output_to_value_info(m.graph)
onnx.save(m, args.out_file)