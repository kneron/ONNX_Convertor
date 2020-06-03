"""Frontend for exporting Caffe graph to ONNX graph
"""
import os.path
import logging
import importlib
import numpy as np
import onnx as O
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from . import helper
from .exceptions import FeatureNotImplemented, OnnxNotSupport
from . import layers

# Please check the README.md for assumptions and modifications.

class CaffeFrontend(object):
  """Keras frontend for Keras
  """
  def __init__(self):
    self.logger = logging.getLogger("onnx-caffe")
    self.logger.setLevel(logging.DEBUG)
    # Keras and Onnx model
    self.cmodel = None
    self.cproto = None
    self.struct = None
    self.omodel = None
    # Helper model attributes
    self.boundaries = set()
    self.outputs = {}
    self.ops= set()
    # True model attributes
    self.values_in = []
    self.values_out = []
    self.node_list = []
    self.value_info = []

  def loadFromFile(self, cppath, cmpath):
    """Read the keras model file and prepare the path for output Onnx model.

    # Arguments:
      cppath: path to the prototxt file.
      cmpath: path to the caffemodel file.
    """
    (dirname, basename) = os.path.split(cmpath)
    self.name = basename.rpartition('.')[0]
    self.opath = os.path.join(dirname, self.name+".onnx")
    self.cmodel = caffe.Net(cppath, cmpath, caffe.TEST)
    self.input_name = None
    for layer_idx, layer in enumerate(self.cmodel.layers):
      # Check for assumptions
      layer_name = self.cmodel._layer_names[layer_idx]
      self.logger.debug("{}: {} ({}) weight: {}".format(layer_idx, layer_name, layer.type, len(layer.blobs)))
    self.logger.debug("Total layer count: {}".format(len(self.cmodel.layers)))
    # net = self.cmodel
    # tops = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in range(0, len(net._blob_names))]

    parsible_net = caffe_pb2.NetParameter()
    text_format.Merge(open(cppath).read(), parsible_net)
    self.cproto = helper.getModelProto(parsible_net)
    self.struct = helper.reconstructNet(parsible_net)
    if self.cmodel._layer_names[0] in self.cproto:
      self.input_name = self.cmodel._layer_names[0]
    else:
      self.input_name = 'data'
    #print("TEST OUTPUT", self.cmodel.blobs['relu1'])

  def loadFromModel(self, model):
    """Read the keras model directly and prepare the path for output Onnx model

    # Arguments:
      model: the Keras model.
    """
    self.cmodel = model
    self.name = 'converted_model'
    self.opath = self.name + '.onnx'

  def saveToFile(self, path=None):
    """Save the Keras model to an onnx file.

    Argument:
      path: the path to save the output file. Default path depends on how you
            loaded the model. If you load the model using loadFromFile, the
            converted file will be under the same folder of the source model,
            with the same name but different suffix. If you load the model
            using loadKerasModule, the path will be the current folder with the
            file name as model name with '.onnx'.
    """
    if path is not None:
      self.opath = path
    O.save(self.omodel, self.opath)

  def convertToOnnx(self):
    """Convert the Caffe model to onnx model.

    # Return:
      It will return an Onnx model
    """
    # Check the caffe model
    if self.cmodel is None:
      raise ValueError("Need to load a caffe model before conversion")
    # Load all layers' converters
    converters = layers
    # Check all layers in Caffe
    for layer_idx, layer in enumerate(self.cmodel.layers):
      if layer.type == 'Scale' or layer.type == 'Split':
        continue
      # Check for assumptions
      layer_name = self.cmodel._layer_names[layer_idx]
      if layer_name not in self.cproto and layer_name == 'input':
        layer_name = self.input_name
      layer_type = layer.type
      self.logger.debug("Processing layer %s(%s)", layer_name, layer_type)
      proto = []
      blob_name = None
      if layer_name in self.cproto:
        proto = self.cproto[layer_name]
        if len(proto.top) > 1:
          self.logger.warning("This inplace layer has more than one top layer")
        else:
          blobs_name = proto.top[0]
      blob = []
      if layer_name in self.cmodel.blobs:
        blob = self.cmodel.blobs[layer_name]
      else:
        blob_name = proto.bottom[0]
        if blob_name in self.cmodel.blobs:
          blob = self.cmodel.blobs[blob_name]
      self.ops.add(layer_type)

      # Prepare output for SSA
      outname = layer_name + '_out'
      if layer_type == 'Python':
        self.logger.warning("Python layer may has wrong output shape")
      if outname in self.boundaries:
        raise FeatureNotImplemented("Use before assign: " + outname)
      if layer_type == 'Input':
        # Input layer
        self.outputs[outname] = self.cmodel.blobs[self.input_name].data.shape
      elif proto and proto.top == proto.bottom:
        top_layer = proto.top
        self.outputs[outname] = self.cmodel.blobs[top_layer[0]].data.shape
      else:
        if blobs_name not in self.cmodel.blobs:
          self.logger.warning("Cannot find blob: {}".format(blobs_name))
          quit(1)
        else:
          shape = self.cmodel.blobs[blobs_name].data.shape
        self.outputs[outname] = shape
      self.logger.debug("Output shape: %s", str(self.outputs[outname]))
      self.boundaries.add(outname)

      ###############
      # Input layer #
      ###############
      if (layer_type == "Input"):
        helper.dtype = helper.convertCaffeType()
        in_var = O.helper.make_tensor_value_info(
          outname,
          helper.dtype,
          self.cmodel.blobs[layer_name].data.shape)
        #print(in_var)
        self.values_in.append(in_var)
        del self.outputs[outname]
      #################
      # General cases #
      #################
      else:
        try:
          Converter = getattr(converters, layer_type)
        except AttributeError:
          Converter = converters.Mystery
        inputs = self.getNodeInputs(proto, self.struct)
        if layer_type == 'BatchNorm':
          if self.cmodel.layers[layer_idx+1].type == 'Scale':
            scale_layer = self.cmodel.layers[layer_idx+1]
          else:
            scale_layer = self.cmodel.layers[layer_idx+4]
          converter = Converter(inputs, outname, [layer, scale_layer], proto, blob)
        else:
          lastblob = self.getLastBlob(proto, self.cmodel.blobs)
          converter = Converter(inputs, outname, layer, proto, lastblob)
        nodes, value_infos = converter.generate()
        self.node_list += nodes
        self.value_info += value_infos

      # Save the output channels for future use
      if (proto and proto.bottom != proto.top and blobs_name in self.cmodel.blobs and len(self.cmodel.blobs[blobs_name].data.shape) == 4):
        helper.channel = self.cmodel.blobs[blobs_name].data.shape[1]
        self.logger.debug("Update channel number buffer to %d", helper.channel)

    # Construct output tensors
    for output in self.outputs:
      shape = np.array(self.outputs[output]).tolist()
      out_var = O.helper.make_tensor_value_info(
        name=output,
        elem_type=helper.dtype,
        shape=shape)
      self.values_out.append(out_var)
      # Also construct one for value info
      self.value_info.append(out_var)

    # Now construct the graph
    graph_def = O.helper.make_graph(
      self.node_list,
      self.name + '_onnx',
      self.values_in,
      self.values_out,
      value_info=self.value_info
      )
    # Create the model (ModelProto)
    self.omodel = O.helper.make_model(graph_def, producer_name='Kneron')
    self.logger.debug("Unknown layer types: {}".format(helper.unknown_types))
    return self.omodel

  # Helper inside a function
  def getNodeInputs(self, cnode, dic):
    """Get the input nodes of the given node
    """
    #.bottom
    inputs = []
    if (dic.get(cnode.name) != None and len(dic.get(cnode.name)) != 0):
      for bottom in dic.get(cnode.name):
        input_name = bottom.name + '_out'
        self.boundaries = self.boundaries | {input_name}
        if input_name in self.outputs:
          out = [int(i) for i in self.outputs[input_name]]
          info = O.helper.make_tensor_value_info(
            input_name,
            helper.dtype,
            out)
          self.value_info.append(info)
          del self.outputs[input_name]
        inputs.append(input_name)
    else:
      for bottom in cnode.bottom:
        input_name = bottom + '_out'
        self.boundaries = self.boundaries | {input_name}
        if input_name in self.outputs:
          out = [int(i) for i in self.outputs[input_name]]
          info = O.helper.make_tensor_value_info(
            input_name,
            helper.dtype,
            out)
          self.value_info.append(info)
          del self.outputs[input_name]
        inputs.append(input_name)
    return inputs

  def getLastBlob(self, cnode, blobs):
    return blobs[cnode.bottom[0]]
