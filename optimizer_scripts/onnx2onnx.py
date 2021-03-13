import onnx
import onnx.utils
from onnx import optimizer
import sys
import argparse
import json

from tools import eliminating
from tools import fusing
from tools import replacing
from tools import other
from tools import special
from tools import combo
# from tools import temp

# Main process
# Argument parser
parser = argparse.ArgumentParser(description="Optimize an ONNX model for Kneron compiler")
parser.add_argument('in_file', help='input ONNX FILE')
parser.add_argument('-o', '--output', dest='out_file', type=str, help="ouput ONNX FILE")
parser.add_argument('--bgr', action='store_true', default=False, help="set if the model is trained in BGR mode")
parser.add_argument('--norm', action='store_true', default=False, help="set if you have the input -0.5~0.5")
parser.add_argument('--split-convtranspose', dest='split_convtranspose', action='store_true', default=False,
                    help="set if you want to split ConvTranspose into Conv and special Upsample")
parser.add_argument('--add-bn-on-skip', dest='bn_on_skip', action='store_true', default=False,
                    help="set if you only want to add BN on skip branches")
parser.add_argument('--add-bn', dest='bn_before_add', action='store_true', default=False,
                    help="set if you want to add BN before Add")
parser.add_argument('-t', '--eliminate-tail-unsupported', dest='eliminate_tail', action='store_true', default=False,
                    help='whether remove the last unsupported node for hardware')
parser.add_argument('--no-bn-fusion', dest='disable_fuse_bn', action='store_true', default=False,
                    help="set if you have met errors which related to inferenced shape mismatch. This option will prevent fusing BatchNormailization into Conv.")
parser.add_argument("-j", "--quantization-json", dest = "quantization_json", 
                    help = "provide quantization info json for further update")

args = parser.parse_args()

if args.out_file is None:
    outfile = args.in_file[:-5] + "_polished.onnx"
else:
    outfile = args.out_file

quantization_info = None 
if args.quantization_json is None:
    print("No quantization json provided")
else:
    with open(args.quantization_json, "r") as load_f:
        quantization_info = json.load(load_f)
        print("load quantization info successfully!")

# onnx Polish model includes:
#    -- nop
#    -- eliminate_identity
#    -- eliminate_nop_transpose
#    -- eliminate_nop_pad
#    -- eliminate_unused_initializer
#    -- fuse_consecutive_squeezes
#    -- fuse_consecutive_transposes
#    -- fuse_add_bias_into_conv
#    -- fuse_transpose_into_gemm

# Basic model organize
m = onnx.load(args.in_file)
# temp.weight_broadcast(m.graph)
m = combo.preprocess(m, args.disable_fuse_bn)
# temp.fuse_bias_in_consecutive_1x1_conv(m.graph)

# Add BN on skip branch
if args.bn_on_skip:
    other.add_bn_on_skip_branch(m.graph, quantization_info)
elif args.bn_before_add:
    other.add_bn_before_add(m.graph, quantization_info)
    other.add_bn_before_activation(m.graph, quantization_info)
# Split deconv
if args.split_convtranspose:
    other.split_ConvTranspose(m)

# My optimization
m = combo.common_optimization(m, quantization_info)
# Special options
if args.bgr:
    special.change_input_from_bgr_to_rgb(m)
if args.norm:
    special.add_0_5_to_normalized_input(m)

# Remove useless last node
if args.eliminate_tail:
    eliminating.remove_useless_last_nodes(m.graph)

# Postprocessing
m = combo.postprocess(m)
onnx.save(m, outfile)

quantization_info_out_dir = outfile[:-5] + "_user_config.json"
with open (quantization_info_out_dir, "w") as f:
    json.dump(quantization_info, f, indent = 1)
    print("Optimized Qunatization information saved")

