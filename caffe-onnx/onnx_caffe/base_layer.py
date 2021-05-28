"""Base layer class
"""
import logging

class Layer:
  def __init__(self, inputs, outname, layer, proto, blob):
    self.inputs = inputs
    self.outputs = [outname]
    self.layer = layer
    self.name = proto.name
    self.proto = proto
    self.blob = blob
    self.logger = logging.getLogger("onnx-caffe")
  def generate(self):
    """Generate the nodes according to the original layer

    # Return value
    node_list   : a list of nodes generated
    value_infos : value_infos between nodes
    """
    return [], []