"""Helper function for constructing the frontend
"""

import logging
import numpy as np
import onnx as O
from onnx import TensorProto
from .exceptions import OnnxNotSupport

# Logger options and flags
logger = logging.getLogger("onnx-keras")
logger.setLevel(logging.DEBUG)
batchReplace = False
custom_name2type = dict()
custom_type2opid = dict()

# Global variables
data_format = None
compatibility = False
dtype = int(TensorProto.FLOAT)
warning_msgs = []
known_tensors = dict()
opid_counter = 0
final_output_change = []
is_sequential = False
duplicate_weights = False
RNN_start = False
RNN_start_node = None

def set_compatibility(enable_compatibility):
  global compatibility
  compatibility = enable_compatibility

def set_duplicate_weights(enable_duplicate):
  """If duplicate_weights is set, all weights should have different names.
  """
  global duplicate_weights
  duplicate_weights = enable_duplicate

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

def warning_once(msg):
  """Check the msg. If duplicated, do not output
  """
  global warning_msgs
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

def convertKerasType(dtype):
  """Convert a numpy type into a tensorflow type
  """
  if dtype not in O.mapping.NP_TYPE_TO_TENSOR_TYPE:
    logger.warning("Unknown data type %s is treated as float", str(dtype))
    dtype = np.dtype('float32')
  return O.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

def formatShape(shape_in):
  """Format the shape before using.
  Change numpy array into list. Remove None. Change data into integer type.
  """
  global batchReplace
  shape = np.array(shape_in).tolist()
  if shape[0] is None:
    if not batchReplace:
      logger.info('Replace None batch size with 1.')
      batchReplace = True
    shape[0] = 1
  for i in range(len(shape)):
    if shape[i] is None:
      raise OnnxNotSupport("Dimension none in shape " + str(shape))
    else:
      shape[i] = int(shape[i])
  return shape

def convertShape(shape_in):
  """ Replace Convert the shape basing on data_format. Convert None to 200s.
  """
  global RNN_start
  shape_in = formatShape(shape_in)
  shape_size = len(shape_in)
  # If the input only has 2 dimensions, do nothing.
  if shape_size < 3:
    return shape_in
  # If the data format is channnels first, do nothing
  if data_format == 'channels_first':
    return shape_in
  # If the shape has more than 2 dimensions and channel last, transpose
  shape_out = []
  shape_out.append(shape_in[0])
  index_so_far = 1
  if RNN_start:
    # For RCNN, sequence number should also be ignored
    shape_out.append(shape_in[index_so_far])
    index_so_far += 1
  # Add the last dimension - channel
  shape_out.append(shape_in[shape_size - 1])
  # Append the left dimensions.
  start = index_so_far
  end = shape_size -1
  for i in range(start, end):
    shape_out.append(shape_in[i])
  return shape_out

def getPadding(size, kernel_size, strides):
  """ Calculate the padding array for same padding in the Tensorflow fashion.
  See https://www.tensorflow.org/api_guides/python/nn#Convolution for more.
  """
  if size[0] % strides[0] == 0:
    pad_h = max(kernel_size[0] - strides[0], 0)
  else:
    pad_h = max(kernel_size[0] - (size[0] % strides[0]), 0)
  if size[1] % strides[1] == 0:
    pad_w = max(kernel_size[1] - strides[1], 0)
  else:
    pad_w = max(kernel_size[1] - (size[1] % strides[1]), 0)
  return [pad_h//2, pad_w//2, pad_h-pad_h//2, pad_w-pad_w//2]

def getConstantNodeByName(tensor_name, weight=None):
  """Get the existing constant node. Otherwise, create it.
  """
  global known_tensors
  if tensor_name in known_tensors:
    logger.debug(tensor_name + " value_info is reused.")
    return [], []
  else:
    if weight is None:
      raise ValueError("Unexpected None value")
    nodes, values = constructConstantNode(tensor_name, weight)
    known_tensors[tensor_name] = weight.shape
    return nodes, values

def constructConstantNode(name, data, output=None):
  """Construct a constant node for weights and other uses.

  # Arguments:
    name:   Name of the node. Usually a node name with usage,
            e.g."conv0_weight"
    data:   The data in numpy format
    output: The output name of current node. By default, it is the current
            node name.
  """
  if data is None:
    raise ValueError("data param cannot be None.")
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

def relu6(x):
  """A fake function as a custom layer
  """
  return x