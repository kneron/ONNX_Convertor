import logging

class FeatureNotImplemented(Exception):
  """Exception raised for not implenented features

    Attributes:
        name -- not implemented feature name
  """
  def __init__(self, name):
    logger = logging.getLogger('onnx-keras')
    logger.error('Not implemented feature: %s', str(name))
    self.name = name
  def __str__(self):
    return (str(self.name) + " is not implemented")

class OnnxNotSupport(Exception):
  """Exception raised for something which is not supported by Onnx.
  """
  def __init__(self, name):
    logger = logging.getLogger('onnx-keras')
    logger.error('Feature is not supported by Onnx: %s', str(name))
    self.name = name
  def __str__(self):
    return (str(self.name) + " is not supported by Onnx.")