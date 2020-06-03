import sys
from onnx_caffe import frontend
import argparse
import logging

#logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser(description='Convert a caffe model into an onnx file.')
#parser.add_argument('kfile', metavar='KerasFile', help='an input hdf5 file')
parser.add_argument('-n', metavar='prototxt', help='an input prototxt file')
parser.add_argument('-w', metavar='caffemodel', help='an input caffemodel file')
parser.add_argument('-o', '--output', dest='ofile', type=str, default="model.onnx", help='the output onnx file')
parser.add_argument('-c', '--custom', dest='custom', type=str, default=None, help='the customized layers definition file')
parser.add_argument('-D', '--debug', action='store_true', default=False, help='whether do various optimizations')
args = parser.parse_args()

# If in debug mode, output debug message
if args.debug:
  logging.basicConfig(level=logging.DEBUG)

# Convert it
converter = frontend.CaffeFrontend()
converter.loadFromFile(args.n, args.w)
onnx_model = converter.convertToOnnx()
converter.saveToFile(args.ofile)
