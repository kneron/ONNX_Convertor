import onnx
import onnx.utils
from onnx import optimizer
import sys
import argparse

from tools import eliminating
from tools import fusing
from tools import replacing
from tools import other
from tools import special
from tools import combo
# from tools import temp

def onnx2onnx_flow(m: onnx.ModelProto, disable_fuse_bn=False, bn_on_skip=False, bn_before_add=False, bgr=False, norm=False, rgba2yynn=False, eliminate_tail=False) -> onnx.ModelProto:
    """Optimize the onnx.

    Args:
        m (ModelProto): the input onnx ModelProto
        disable_fuse_bn (bool, optional): do not fuse BN into Conv. Defaults to False.
        bn_on_skip (bool, optional): add BN operator on skip branches. Defaults to False.
        bn_before_add (bool, optional): add BN before Add node on every branches. Defaults to False.
        bgr (bool, optional): add an Conv layer to convert rgb input to bgr. Defaults to False.
        norm (bool, optional): add an Conv layer to add 0.5 tp the input. Defaults to False.
        rgba2yynn (bool, optional): add an Conv layer to convert rgb input to yynn . Defaults to False.
        eliminate_tail (bool, optional): remove the trailing NPU unsupported nodes. Defaults to False.

    Returns:
        ModelProto: the optimized onnx model object.
    """
    # temp.weight_broadcast(m.graph)
    m = combo.preprocess(m, disable_fuse_bn)
    # temp.fuse_bias_in_consecutive_1x1_conv(m.graph)

    # Add BN on skip branch
    if bn_on_skip:
        other.add_bn_on_skip_branch(m.graph)
    elif bn_before_add:
        other.add_bn_before_add(m.graph)
        other.add_bn_before_activation(m.graph)

    # My optimization
    m = combo.common_optimization(m)
    # Special options
    if bgr:
        special.change_input_from_bgr_to_rgb(m)
    if norm:
        special.add_0_5_to_normalized_input(m)
    if rgba2yynn:
        special.add_rgb2yynn_node(m)

    # Remove useless last node
    if eliminate_tail:
        eliminating.remove_useless_last_nodes(m.graph)

    # Postprocessing
    m = combo.postprocess(m)
    return m

# Main process
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Optimize an ONNX model for Kneron compiler")
    parser.add_argument('in_file', help='input ONNX FILE')
    parser.add_argument('-o', '--output', dest='out_file', type=str, help="ouput ONNX FILE")
    parser.add_argument('--bgr', action='store_true', default=False, help="set if the model is trained in BGR mode")
    parser.add_argument('--norm', action='store_true', default=False, help="set if you have the input -0.5~0.5")
    parser.add_argument('--rgba2yynn', action='store_true', default=False, help="set if the model has yynn input but you want to take rgba images")
    parser.add_argument('--add-bn-on-skip', dest='bn_on_skip', action='store_true', default=False,
                        help="set if you only want to add BN on skip branches")
    parser.add_argument('--add-bn', dest='bn_before_add', action='store_true', default=False,
                        help="set if you want to add BN before Add")
    parser.add_argument('-t', '--eliminate-tail-unsupported', dest='eliminate_tail', action='store_true', default=False,
                        help='whether remove the last unsupported node for hardware')
    parser.add_argument('--no-bn-fusion', dest='disable_fuse_bn', action='store_true', default=False,
                        help="set if you have met errors which related to inferenced shape mismatch. This option will prevent fusing BatchNormailization into Conv.")
    args = parser.parse_args()

    if args.out_file is None:
        outfile = args.in_file[:-5] + "_polished.onnx"
    else:
        outfile = args.out_file

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

    m = onnx2onnx_flow(m, args.disable_fuse_bn, args.bn_on_skip, args.bn_before_add, args.bgr, args.norm, args.rgba2yynn, args.eliminate_tail)

    onnx.save(m, outfile)
