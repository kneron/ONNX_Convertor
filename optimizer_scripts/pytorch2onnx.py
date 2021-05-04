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
from pytorch_exported_onnx_preprocess import torch_exported_onnx_flow

# Debug use
# logging.basicConfig(level=logging.DEBUG)

######################################
#  Generate a prototype onnx         #
######################################

parser = argparse.ArgumentParser(description="Optimize a Pytorch generated model for Kneron compiler")
parser.add_argument('in_file', help='input ONNX or PTH FILE')
parser.add_argument('out_file', help="ouput ONNX FILE")
parser.add_argument('--input-size', dest='input_size', nargs=3,
                    help='if you using pth, please use this argument to set up the input size of the model. It should be in \'CH H W\' format, e.g. \'--input-size 3 256 512\'.')
parser.add_argument('--no-bn-fusion', dest='disable_fuse_bn', action='store_true', default=False,
                    help="set if you have met errors which related to inferenced shape mismatch. This option will prevent fusing BatchNormailization into Conv.")

args = parser.parse_args()

if len(args.in_file) <= 4:
    # When the filename is too short.
    logging.error("Invalid input file: {}".format(args.in_file))
    exit(1)
elif args.in_file[-4:] == '.pth':
    # Pytorch pth case
    logging.warning("Converting from pth to onnx is not recommended.")
    onnx_in = args.out_file
    # Import pytorch libraries
    from torch.autograd import Variable
    import torch
    import torch.onnx
    # import torchvision
    # Standard ImageNet input - 3 channels, 224x224.
    # Values don't matter as we care about network structure.
    # But they can also be real inputs.
    if args.input_size is None:
        logging.error("\'--input-size\' is required for the pth input file.")
        exit(1)
    dummy_input = Variable(torch.randn(1, int(args.input_size[0]), int(args.input_size[1]), int(args.input_size[2])))
    # Obtain your model, it can be also constructed in your script explicitly.
    model = torch.load(sys.argv[1], map_location='cpu')
    # model = torchvision.models.resnet34(pretrained=True)
    # Invoke export.
    # torch.save(model, "resnet34.pth")
    torch.onnx.export(model, dummy_input, args.out_file, opset_version=11)
elif args.in_file[-4:] == 'onnx':
    onnx_in = args.in_file
else:
    # When the file is neither an onnx or a pytorch pth.
    logging.error("Invalid input file: {}".format(args.in_file))
    exit(1)

onnx_out = args.out_file

######################################
#  Optimize onnx                     #
######################################

m = onnx.load(onnx_in)

m = torch_exported_onnx_flow(m, args.disable_fuse_bn)

onnx.save(m, onnx_out)
