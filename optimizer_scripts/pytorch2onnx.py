from torch.autograd import Variable
import torch
import torch.onnx
import torchvision
import onnx
import onnx.utils
from onnx import optimizer
import sys
import numpy as np
import struct
import logging

from tools import eliminating
from tools import fusing
from tools import replacing
from tools import other
from tools import combo

#logging.basicConfig(level=logging.DEBUG)
######################################
#  Generate a prototype onnx         #
######################################
if len(sys.argv) != 3 and len(sys.argv) != 6:
    print("python pytorch2onnx.py PTH CH H W ONNX")
    print("python pytorch2onnx.py ONNX_IN ONNX_OUT")
    exit(1)

if len(sys.argv) == 6:
    logging.warning("Converting from pth to onnx is not recommended.")
    onnx_in = sys.argv[5]
    onnx_out = sys.argv[5]
    # Standard ImageNet input - 3 channels, 224x224.
    # Values don't matter as we care about network structure.
    # But they can also be real inputs.
    dummy_input = Variable(torch.randn(
        1, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
    # Obtain your model, it can be also constructed in your script explicitly.
    model = torch.load(sys.argv[1], map_location='cpu')
    # model = torchvision.models.resnet34(pretrained=True)
    # Invoke export.
    # torch.save(model, "resnet34.pth")
    if torch.__version__ < '1.3.0':
        torch.onnx.export(model, dummy_input, sys.argv[5])
        torch.onnx.export(model, dummy_input,
                          sys.argv[5][:-5] + "_backup.onnx")
    else:
        torch.onnx.export(model, dummy_input,
                          sys.argv[5], keep_initializers_as_inputs=True)
        torch.onnx.export(
            model, dummy_input, sys.argv[5][:-5] + "_backup.onnx", keep_initializers_as_inputs=True)
else:
    onnx_in = sys.argv[1]
    onnx_out = sys.argv[2]

######################################
#  Optimize onnx                     #
######################################

m = onnx.load(onnx_in)

other.pytorch_check_initializer_as_input(m.graph)
m = combo.preprocess(m)
m = combo.pytorch_constant_folding(m)

m = combo.common_optimization(m)

m = combo.postprocess(m)
onnx.save(m, onnx_out)
