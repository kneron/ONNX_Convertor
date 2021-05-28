"""Helper function for constructing the frontend
"""

import logging
import numpy as np
import onnx as O
from onnx import mapping

# Logger options and flags
logger = logging.getLogger("onnx-caffe")
batchReplace = False
noneDataFormat = False

# Global variables
channel = None
dtype = np.dtype('float32')
warning_msgs = []
custom_name2type = dict()
custom_type2opid = dict()
opid_counter = 0
unknown_types = set()

def warning_once(msg):
  """Check the msg. If duplicated, do not output
  """
  global warning_msgs
  global logger
  if msg not in warning_msgs:
    warning_msgs.append(msg)
    logger.warning(msg)

def getKerasLayerType(layer):
  """Get the keras layer type name as a string
  """
  return str(type(layer)).split('.')[-1][:-2]

def hasActivation(activation):
  """Check if this activation does something
  """
  return activation.__name__ != 'linear'

def convertCaffeType():
  dtype = np.dtype('float32')
  return mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

def convertKerasType(dtype):
  """Convert a numpy type into a tensorflow type
  """
  if dtype not in mapping.NP_TYPE_TO_TENSOR_TYPE:
    logger.warning("Unknown data type %s is treated as float", str(dtype))
    dtype = np.dtype('float32')
  return mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

def getPadding(size, kernel_size, strides, pad):
  """ Calculate the padding array for same padding in the Tensorflow fashion.
  See https://www.tensorflow.org/api_guides/python/nn#Convolution for more.
  """
  pads = [0, 0, 0, 0]
  if ((size[0] - kernel_size[0]) % strides[0] == 0):
    pads[0] = pad
    pads[2] = pad
  else:
    if size[0] % strides[0] == 0:
      pad_h = max(kernel_size[0] - strides[0], 0)
    else:
      pad_h = max(kernel_size[0] - (size[0] % strides[0]), 0)
    pads[0] = pad + pad_h // 2
    pads[2] = pad + pad_h - pad_h // 2
  if ((size[1] - kernel_size[1]) % strides[1] == 0):
    pads[1] = pad
    pads[3] = pad
  else:
    if size[1] % strides[1] == 0:
      pad_w = max(kernel_size[1] - strides[1], 0)
    else:
      pad_w = max(kernel_size[1] - (size[1] % strides[1]), 0)
    pads[1] = pad + pad_w // 2
    pads[3] = pad + pad_w - pad_w // 2
  return pads

def constructConstantNode(name, data, output=None):
  """Construct a constant node for weights and other uses.

  # Arguments:
    name:   Name of the node. Usually a node name with usage,
            e.g."conv0_weight"
    data:   The data in numpy format
    output: The output name of current node. By default, it is the current
            node name.
  """
  tensor = O.helper.make_tensor(
    name + 'tensor',
    convertKerasType(data.dtype),
    data.shape,
    data.ravel())
  if output is None:
    output = name
  node = O.helper.make_node(
    "Constant",
    [],
    [output],
    name=name,
    value=tensor)
  node_list = [node]
  info = O.helper.make_tensor_value_info(
    name,
    convertKerasType(data.dtype),
    data.shape
  )
  value_infos = [info]
  return node_list, value_infos

def getModelProto(proto):
  result = dict()
  if len(proto.layer) == 0:
    layers = proto.layers
  else:
    layers = proto.layer
  for layer in layers:
    result[layer.name] = layer
  return result

def reconstructNet(proto):
  blobs = dict()
  layer_map = dict()
  result = dict()

  for layer in proto.layer:
    layer_map[layer.name] = layer

  #reconstruct the graph
  for layer in proto.layer:
    if layer.name in result:
      logger.warning("Layer with name " + layer.name + " appears at least twice in the model.")
      logger.warning("We will pickup the later one instead of using all of them.")
    result[layer.name] = list()
    for bottom in layer.bottom:
      if bottom in blobs:
        result[layer.name].append(blobs[bottom])
    for top in layer.top:
      blobs[top] = layer

  #remove discarded layers
  discarded_layer = ['Scale', 'Split']

  for layer_name, pre_layers in result.items():
    new_list = list()
    for layer in pre_layers:
      if layer.type in discarded_layer:
        new_list = new_list + result[layer.name]
      else:
        new_list.append(layer)
    result[layer_name] = new_list

  return result
  #print(result['conv1_1/conv'])

def set_custom_layer(custom_list):
  """From the list construct a dictionary for custom layers
  """
  global custom_name2type
  global custom_type2opid
  global opid_counter
  for custom_layer in custom_list:
    opid = opid_counter
    opid_counter += 1
    custom_type2opid[custom_layer["layer_type"]] = opid
    for custom_name in custom_layer["layer_names"]:
      custom_name2type[custom_name] = custom_layer["layer_type"]

