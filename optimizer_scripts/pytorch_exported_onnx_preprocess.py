import onnx
import onnx.utils
from onnx import optimizer
import sys
import numpy as np
import struct
import logging
import argparse

from tools import eliminating
from tools import fusing
from tools import replacing
from tools import other
from tools import combo
from tools import special

# Define general pytorch exported onnx optimize process
def torch_exported_onnx_flow(m: onnx.ModelProto, disable_fuse_bn=False) -> onnx.ModelProto:
    """Optimize the Pytorch exported onnx.

    Args:
        m (ModelProto): the input onnx model
        disable_fuse_bn (bool, optional): do not fuse BN into Conv. Defaults to False.

    Returns:
        ModelProto: the optimized onnx model
    """
    m = combo.preprocess(m, disable_fuse_bn, duplicate_shared_weights=False)
    m = combo.pytorch_constant_folding(m)
    m = combo.common_optimization(m)
    m = combo.postprocess(m)

    return m


# Main Process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a Pytorch generated model for Kneron compiler")
    parser.add_argument('in_file', help='input ONNX')
    parser.add_argument('out_file', help="ouput ONNX FILE")
    parser.add_argument('--log', default='info', type=str, help="set log level")
    parser.add_argument('--no-bn-fusion', dest='disable_fuse_bn', action='store_true', default=False,
                        help="set if you have met errors which related to inferenced shape mismatch. This option will prevent fusing BatchNormailization into Conv.")
    parser.add_argument('--input', dest='input_change', type=str, nargs='+', help="change input shape (e.g. --input 'input_0 1 3 224 224')")
    parser.add_argument('--output', dest='output_change', type=str, nargs='+', help="change output shape (e.g. --output 'input_0 1 3 224 224')")

    args = parser.parse_args()

    if args.log == 'warning':
        logging.basicConfig(level=logging.WARN)
    elif args.log == 'debug':
        logging.basicConfig(level=logging.DEBUG)
    elif args.log == 'error':
        logging.basicConfig(level=logging.ERROR)
    elif args.log == 'info':
        logging.basicConfig(level=logging.INFO)
    else:
        print(f"Invalid log level: {args.log}")
        logging.basicConfig(level=logging.INFO)

    if len(args.in_file) <= 4:
        # When the filename is too short.
        logging.error("Invalid input file: {}".format(args.in_file))
        exit(1)
    elif args.in_file[-4:] == 'onnx':
        onnx_in = args.in_file
    else:
        # When the file is not an onnx file.
        logging.error("Invalid input file: {}".format(args.in_file))
        exit(1)

    onnx_out = args.out_file

    ######################################
    #  Optimize onnx                     #
    ######################################

    m = onnx.load(onnx_in)

    # Change input and output shapes as requested
    if args.input_change is not None:
        other.change_input_shape(m.graph, args.input_change)
    if args.output_change is not None:
        other.change_output_shape(m.graph, args.output_change)

    m = torch_exported_onnx_flow(m, args.disable_fuse_bn)

    onnx.save(m, onnx_out)
