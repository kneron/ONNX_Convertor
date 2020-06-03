from . import helper

class TreeNode(object):
  """Intermedia structure for model graph
  """
  def __init__(self, klayer=None, prefix='', postfix='', pos=0):
    """Initialize a tree node object

    # Arguments:
      klayer: The layer defined in keras.
      prefix: The prefix inherited from the parent layer. No prefix by default.
      postfix: Usually the position name of the node for the shared layer.
      pos: The position of the current node among all duplicated shared layers.
    """
    self.inputs = []
    self.outputs = []
    self.pos = pos
    self.new_w = None
    self.new_b = None
    self.extra_attr = None
    self.klayer = klayer
    if klayer is None:
      return
    self.name = prefix + klayer.name + postfix
    self.type = helper.getKerasLayerType(self.klayer)
    # Check data_format
    if hasattr(klayer, "data_format"):
      if helper.data_format is None:
        helper.data_format = klayer.data_format
      elif helper.data_format != klayer.data_format:
        raise ValueError("You cannot have two different data_format in the same model")

  def print_info(self):
    """Print out the infomation of the current Node
    """
    helper.logger.debug(self.name)
    for input_tensor in self.inputs:
      helper.logger.debug('\t<- ' + input_tensor.name)
    for output_tensor in self.outputs:
      helper.logger.debug('\t-> ' + output_tensor.name)

  def get_output_tensors(self):
    """Get all the output keras tensors of the current node
    """
    if self.klayer is None:
      return [self.outputs[0].tensor]
    return self.klayer._inbound_nodes[self.pos].output_tensors

  def get_input_tensors(self):
    """Get all the input keras tensors of the current node
    """
    if self.klayer is None:
      return [self.outputs[0].tensor]
    return self.klayer._inbound_nodes[self.pos].input_tensors

  def replace_input(self, new_input, old_input):
    """Replace one old input with the new input
    """
    for i in range(len(self.inputs)):
      if self.inputs[i] == old_input:
        self.inputs[i] = new_input
        return
    assert False, "Cannot find the specific input"

  def check_input_ready(self, converted_tree_tensor):
    """Check whether all the input value infos of the current layer is ready.
    """
    # Input layers depend on nothing and are alway ready.
    if self.type == "InputLayer":
      return True
    # Otherwise, check all the dependencies are ready.
    prepared = True
    for input_value in self.inputs:
      if input_value.name not in converted_tree_tensor:
        prepared = False
    return prepared

class TreeTensor(object):
  """A wrapper of the keras tensor. Used to construct the tree
  """
  def __init__(self, name, tensor=None, creator=None):
    self.name = name
    self.tensor = tensor
    self.input = None
    self.outputs = []
    self.keras_shape = None
    self.shape = None
    if creator is not None:
      self.input = creator

  def replace_output(self, new_outputs, old_output):
    """Replace one old output with the new output
    """
    for i in range(len(self.outputs)):
      if self.outputs[i] == old_output:
        del self.outputs[i]
        self.outputs += new_outputs
        return
    assert False, "Cannot find the specific output"

  def set_shape(self, keras_shape, shape=None):
    keras_shape = helper.formatShape(keras_shape)
    if shape is None:
      shape = helper.convertShape(keras_shape)
    else:
      shape = helper.formatShape(shape)
    self.keras_shape = keras_shape
    self.shape = shape

  def get_shape(self):
    """Get the channel first shape of this specific tensor
    """
    return map(convert2int, list(self.tensor.shape))

def convert2int(x):
  try:
    a = int(x)
  except TypeError:
    a = 1
  return a