"""Converters for core layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
import utils


class Relu(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      relu_node = helper.make_node(
          "Relu",
          name=self.onnx_node_name, 
          inputs=self.previous_onnx_node_names, 
          outputs=[self.onnx_node_name]
      )
      self.node_list.append(relu_node)
      
      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.onnx_node_name, 
          TensorProto.FLOAT, 
          utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
      )
      self.value_infos.append(out_shape_info)

      return self.node_list, self.value_infos, self.weight_node_list
    

class Relu6(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      clip_node = onnx.helper.make_node(
          'Clip',
          inputs=self.previous_onnx_node_names,
          outputs=[self.onnx_node_name],
          min=0.0,
          max=6.0,
          name=self.onnx_node_name
      )
      self.node_list.append(clip_node)
      
      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.onnx_node_name, 
          TensorProto.FLOAT, 
          utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
      )
      self.value_infos.append(out_shape_info)

      return self.node_list, self.value_infos, self.weight_node_list


class LOGISTIC(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      logistic_name = self.onnx_node_name
      logistic_node = helper.make_node( 
          op_type='Sigmoid', 
          inputs=self.previous_onnx_node_names,
          outputs=[logistic_name], 
          name=logistic_name 
      )
      self.node_list.append(logistic_node)

      return self.node_list, self.value_infos, self.weight_node_list

class Softmax(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      softmax_node_name = self.onnx_node_name
      softmax_node = onnx.helper.make_node(
          'Softmax',
          inputs=self.previous_onnx_node_names,
          outputs=[softmax_node_name],
          name=softmax_node_name
      )
      self.node_list.append(softmax_node)

      return self.node_list, self.value_infos, self.weight_node_list


