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

# Debug use
# logging.basicConfig(level=logging.DEBUG)

######################################
#  Generate a prototype onnx         #
######################################

parser = argparse.ArgumentParser(description="Optimize a Pytorch generated model for Kneron compiler")
parser.add_argument('in_file', help='input ONNX')
parser.add_argument('out_file', help="ouput ONNX FILE")
parser.add_argument('--no-bn-fusion', dest='disable_fuse_bn', action='store_true', default=False,
                    help="set if you have met errors which related to inferenced shape mismatch. This option will prevent fusing BatchNormailization into Conv.")
parser.add_argument('--align-corner', dest='align_corner', action='store_true', default=False,
                    help="set if the Upsample nodes in your pytorch model have align_corner set to true.")

args = parser.parse_args()

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

def torch_exported_onnx_flow(m, disable_fuse_bn=False, align_corner=False):
    special.check_onnx_version(m)

    m = combo.preprocess(m, disable_fuse_bn)
    m = combo.pytorch_constant_folding(m)

    m = combo.common_optimization(m)

    m = combo.postprocess(m)

    if align_corner:
        special.set_upsample_mode_to_align_corner(m.graph)
    return m

######################################
#  Optimize onnx                     #
######################################

m = onnx.load(onnx_in)

m = torch_exported_onnx_flow(m. args.disable_fuse_bn, args.align_corner)

onnx.save(m, onnx_out)
