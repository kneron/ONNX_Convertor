"""Converters for core layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
from aact_layers import defused_activation_node_generator
import utils

from tflite.AddOptions import AddOptions
from tflite.MulOptions import MulOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.ActivationFunctionType import ActivationFunctionType


def make_onnx_constant_number(tflite_interpreter, constant_details):
    constant_array = tflite_interpreter.get_tensor(constant_details['index'])

    # make bias onnx node
    tensor_node_name = constant_details['name']
    tensor_node = onnx.helper.make_tensor(
        tensor_node_name,
        TensorProto.FLOAT,
        constant_array.shape,
        constant_array.flatten().tolist()
    )
    
    return tensor_node

def make_onnx_channelwise_constant_number(tflite_interpreter, constant_details):
    constant_array = tflite_interpreter.get_tensor(constant_details['index'])

    # make bias onnx node
    tensor_node_name = constant_details['name']
    tensor_node = onnx.helper.make_tensor(
        tensor_node_name,
        TensorProto.FLOAT,
        [1,constant_array.shape[0],1,1],
        constant_array.flatten().tolist()
    )
    
    return tensor_node

class Add(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_add_parser = AddOptions()
      self.tflite_add_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      prev_node_names = self.input_nodes_name.copy()
      
      for input_idx in range(self.op.InputsLength()):

          node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(input_idx))
          if node_input_detail['shape'].size == 4:
              # it's a op node, do nothing
              continue

          prev_node_name = node_input_detail['name']

          # it's a parameter node, create constant node
          if node_input_detail['shape'].size == 0:
              # add all
              tensor_node = make_onnx_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)
          elif node_input_detail['shape'].size == 1:
              # channelwise add
              tensor_node = make_onnx_channelwise_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)

          prev_node_names.append(prev_node_name)
      

      add_node_name = self.node_name
      add_node = onnx.helper.make_node(
          'Add',
          inputs=prev_node_names,
          outputs=[add_node_name],
          name=add_node_name
      )

      # update tables
      self.node_list.append(add_node)

      # change output node's input_name
      for o_n in self.output_nodes:
          for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
              if o_n_i_n == self.node_name:
                  o_n.input_nodes_name[idx] = self.node_list[-1].name

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_add_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)

class Concatenation(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_concat_parser = ConcatenationOptions()
      self.tflite_concat_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):

      prev_node_names = self.input_nodes_name.copy()

      concat_node_name = self.node_name
      concat_node = onnx.helper.make_node(
          'Concat',
          inputs=prev_node_names,
          outputs=[concat_node_name],
          axis= utils.channel_last_2_channel_first_axis_mapping( [self.tflite_concat_parser.Axis()] )[0],
          name=concat_node_name
      )

      # update tables
      self.node_list.append(concat_node)

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_concat_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)

class Mul(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_mul_parser = MulOptions()
      self.tflite_mul_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):
      prev_node_names = self.input_nodes_name.copy()
      for input_idx in range(self.op.InputsLength()):
          node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(input_idx))

          if node_input_detail['shape'].size == 4:
              # it's a op node, do nothing
              continue

          prev_node_name = node_input_detail['name']

          # create constant node
          if node_input_detail['shape'].size == 0:
              # mul all
              tensor_node = make_onnx_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)
          elif node_input_detail['shape'].size == 1:
              # channelwise mul
              tensor_node = make_onnx_channelwise_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)

          prev_node_names.append(prev_node_name)

      mul_node_name = self.node_name
      mul_node = onnx.helper.make_node(
          'Mul',
          inputs=prev_node_names,
          outputs=[mul_node_name],
          name=mul_node_name
      )

      # update tables
      self.node_list.append(mul_node)

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_mul_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)