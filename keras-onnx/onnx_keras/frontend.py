"""Frontend for exporting Keras graph to ONNX graph
"""
import os.path
import logging
import importlib
import numpy as np
import keras.models as Kmodels
import keras.utils as Kutils
import onnx as O
from onnx import shape_inference
import keras as K
import tensorflow as tf

from . import helper
from . import preprocess
from . import layers
from .exceptions import FeatureNotImplemented, OnnxNotSupport

# Please check the README.md for assumptions and modifications.

class KerasFrontend(object):
  """Keras frontend for Keras
  """
  def __init__(self):
    self.logger = logging.getLogger("onnx-keras")
    self.logger.setLevel(logging.DEBUG)
    # Keras and Onnx model
    self.kmodel = None
    self.omodel = None
    # Helper model attributes
    self.ops = set()
    # True model attributes
    self.values_in = []
    self.values_out = []
    self.node_list = []
    self.value_info = []

  def loadFromFile(self, kpath):
    """Read the keras model file and prepare the path for output Onnx model.

    # Arguments:
      kpath: path to the Keras model(.hdf5) file.
    """
    kpath = os.path.abspath(kpath)
    (dirname, basename) = os.path.split(kpath)
    self.name = basename.rpartition('.')[0]
    self.opath = os.path.join(dirname, self.name+".onnx")
    self.kmodel = Kmodels.load_model(kpath, custom_objects={ 'tf': tf, 'relu6': helper.relu6 })

  def loadFromModel(self, model):
    """Read the keras model directly and prepare the path for output Onnx model

    # Arguments:
      model: the Keras model.
    """
    self.kmodel = model
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

  def convertToOnnx(self, optimize=0, input_shape=None):
    """Convert the Keras model to onnx model.

    # Return:
      It will return an Onnx model
    """
    ## 1. Prepare the environment and the model
    # Check the keras model
    if self.kmodel is None:
      raise ValueError("Need to load a keras model before conversion")
    # Load all layers' converters
    converters = layers

    ## 2. Construct and process the graph
    #  Preprocess and generate a layer tree
    self.logger.info("Preprocessing the graph")
    self.tree, self.input_nodes, self.output_tensors = preprocess.preprocess(self.kmodel, optimize=optimize)
    # Check all the layers in Keras
    self.logger.info("Start processing the graph")
    # Check the data format
    if helper.data_format is None:
      self.logger.warning("There is no data format specified in the model. Assume it is channels last.")
      helper.data_format = 'channels_last'
    if input_shape is not None:
      self.logger.warning("Currently, custom input size is only available for single input size. Mystery node may generate wrong size.")
      input_shape = list(map(int, input_shape))

    ## 3. Use worklist to process all the layers.
    # Initialize some variables for the worklist
    worklist = list(self.tree)
    converted_tree_tensors = dict()
    last_count = len(worklist)
    cur = 0
    while len(worklist) != 0:
      # If we are reaching the end, check whether there is any change in worklist.
      # If changes exist, back to the beginning. Otherwise, exit.
      if cur == len(worklist):
        if cur == last_count:
          self.logger.warning("The nodes listed below are not reachable:")
          for bad_node in worklist:
            self.logger.warning(bad_node.name)
          break
        cur = 0
      node = worklist[cur]
      # If the current layer is still not ready to be processed, process the next one.
      if not node.check_input_ready(converted_tree_tensors):
        cur += 1
        continue
      # Current layer can be processed.
      # Prepare some debug information
      self.logger.debug("Processing layer %s(%s)", node.name, node.type)
      self.ops.add(node.type)

      # Check if the current node has only one output.
      if len(node.outputs) != 1:
        raise FeatureNotImplemented('Operator with more than one outputs')

      ###############
      # Input layer #
      ###############
      # Input layer is special since it has no input nodes.
      # And onnx do not have the input operator but input value info instead.
      if (node.type == "InputLayer"):
        if node.klayer is not None:
          config = node.klayer.get_config()
          helper.dtype = helper.convertKerasType(np.dtype(config['dtype']))
          tree_tensor = node.outputs[0]
          if input_shape is not None:
            if len(input_shape) != len(node.klayer.input_shape):
              raise RuntimeError("Unmatch input shape: expected {}, got {}".format(
                node.klayer.input_shape, input_shape))
            tree_tensor.set_shape(input_shape)
          else:
            tree_tensor.set_shape(node.klayer.input_shape)
        else:
          tree_tensor = node.outputs[0]
          if input_shape is not None:
            if len(input_shape) != len(self.tree[1].klayer.input_shape):
              raise RuntimeError("Unmatch input shape: expected {}, got {}".format(
                self.tree[1].klayer.input_shape, input_shape))
            tree_tensor.set_shape(input_shape)
          else:
            tree_tensor.set_shape(self.tree[1].klayer.input_shape)
        in_var = O.helper.make_tensor_value_info(
          name=node.outputs[0].name,
          elem_type=helper.dtype,
          shape=tree_tensor.shape
        )
        converted_tree_tensors[tree_tensor.name] = tree_tensor
        self.values_in.append(in_var)

      #################
      # General cases #
      #################
      else:
        # Set up converter by layer name
        try:
          Converter = getattr(converters, node.type)
          converter = Converter(node)
        except AttributeError:
          helper.warning_once("OP " + node.type + " is an unknown layer. Using CustomOP layer instead.")
          converter = layers.Lambda(node)
        # Infer the output shape
        node_output_value = converter.setOutputValue()
        self.value_info.append(node_output_value)
        tree_tensor = node.outputs[0]
        converted_tree_tensors[tree_tensor.name] = tree_tensor
        # Convert and append to finished list
        nodes, value_infos = converter.generate()
        self.node_list += nodes
        self.value_info += value_infos

      self.logger.debug("Output shape: %s", str(node.outputs[0].shape))

      # Delete current layer from the worklist. And start from the beginning.
      del worklist[cur]
      cur = 0

    # Construct output tensors
    for output in self.output_tensors:
      # Compare output shape with the shape from the value infos
      shape = output.shape
      if output.name not in converted_tree_tensors:
        raise ValueError("Unknown output tensor: ", output.name)
      if converted_tree_tensors[output.name].shape != shape:
        if output.name in helper.final_output_change:
          shape = converted_tree_tensors[output.name].shape
          self.logger.debug("Ignore " + output.name + " for output shape check")
        raise ValueError("Unmatched output shape: ", converted_tree_tensors[output.name].shape, shape)
      # Generate outputs
      out_var = O.helper.make_tensor_value_info(
        name=output.name,
        elem_type=helper.dtype,
        shape=shape)
      self.values_out.append(out_var)

    # Now construct the graph
    self.logger.debug("Nodes:")
    for node in self.node_list:
      self.logger.debug(node.name)
    graph_def = O.helper.make_graph(
      self.node_list,
      self.name + '_onnx',
      self.values_in,
      self.values_out,
      value_info=self.value_info
      )
    # Create the model (ModelProto)
    self.omodel = O.helper.make_model(graph_def, producer_name='Kneron')
    self.omodel.opset_import[0].version = 9
    # O.checker.check_model(self.omodel)
    self.logger.debug("Conversion Finished. With op: " + str(self.ops))
    return self.omodel
