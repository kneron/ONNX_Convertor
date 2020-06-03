import sys
import argparse
import logging
import onnx_keras
import json

parser = argparse.ArgumentParser(description='Convert a Keras hdf5 file into an onnx file.')
parser.add_argument('kfile', metavar='KerasFile', help='an input hdf5 file')
parser.add_argument('-o', '--output', dest='ofile', type=str, default="output.onnx", help='the output onnx file')
parser.add_argument('-c', '--custom', dest='custom', type=str, default=None, help='the customized layer definition file')
parser.add_argument('-O', '--optimize', nargs='?', dest='optimize', type=int, default=0, const=3, help='set optimization level for Kneron hardware')
parser.add_argument('-D', '--debug', action='store_true', default=False, help='whether do various optimizations')
parser.add_argument('-C', '--compatibility', action='store_true', default=False, help='whether enable compatibility mode')
parser.add_argument('-I', '--input-shape', dest='input_shape', nargs='+', help='give a custom shape which matches the hdf5 model input')
parser.add_argument('--duplicate-shared-weights', action='store_true', dest='duplicate_weights', default=False, help='duplicate shared weights if set')

args = parser.parse_args()

onnx_keras.set_compatibility(args.compatibility)
onnx_keras.set_duplicate_weights(args.duplicate_weights)

# If in debug mode, output debug message
if args.debug:
  logging.basicConfig(level=logging.DEBUG)

# Setup custom layers
if args.custom is not None:
  f = open(args.custom, 'r')
  onnx_keras.set_custom_layer((json.load(f))["layer classes"])

converter = onnx_keras.frontend.KerasFrontend()
converter.loadFromFile(args.kfile)
onnx_model = converter.convertToOnnx(args.optimize, args.input_shape)
converter.saveToFile(args.ofile)
