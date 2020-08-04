"""Converters for core layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
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

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

      self.tflite_add_parser = AddOptions()
      self.tflite_add_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos) 

  def generate(self, op_name__sub_op_name__table):
      prev_node_names = []
      for input_idx in range(self.op_info.InputsLength()):
          node_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(input_idx))

          prev_node_name = node_input_detail['name']

          if node_input_detail['name'] in op_name__sub_op_name__table:
              prev_node_name = op_name__sub_op_name__table[node_input_detail['name']][-1] # last sub node

          # create constant node
          if node_input_detail['shape'].size == 0:
              # add all
              tensor_node = make_onnx_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)
          elif node_input_detail['shape'].size == 1:
              # channelwise add
              tensor_node = make_onnx_channelwise_constant_number(self.tflite_interpreter, node_input_detail)
              self.weight_node_list.append(tensor_node)

          prev_node_names.append(prev_node_name)

      add_node_name = self.onnx_node_name
      add_node = onnx.helper.make_node(
          'Add',
          inputs=prev_node_names,
          outputs=[add_node_name],
          name=add_node_name
      )

      # update tables
      self.node_list.append(add_node)

      activative_op = self.tflite_add_parser.FusedActivationFunction()
      if activative_op == ActivationFunctionType.RELU6:
          clip_name = 'fused_clip_' + self.onnx_node_name
          clip_node = onnx.helper.make_node('Clip',inputs=[self.onnx_node_name],outputs=[clip_name],min=0.0,max=6.0,name=clip_name)
          out_shape_info = onnx.helper.make_tensor_value_info(
              clip_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(clip_node)

      elif activative_op == ActivationFunctionType.RELU:
          relu_name = 'fused_relu_' + self.onnx_node_name
          relu_node = onnx.helper.make_node("Relu",name=relu_name, inputs=[self.onnx_node_name], outputs=[relu_name])
          out_shape_info = onnx.helper.make_tensor_value_info(
              relu_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(relu_node)

      return self.node_list, self.value_infos, self.weight_node_list

class Concatenation(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

      self.tflite_concat_parser = ConcatenationOptions()
      self.tflite_concat_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos) 

  def generate(self,op_name__sub_op_name__table):

      prev_node_names = []
      for input_idx in range(self.op_info.InputsLength()):
          node_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(input_idx))
          prev_node_name = node_input_detail['name']
          if node_input_detail['name'] in op_name__sub_op_name__table:
              prev_node_name = op_name__sub_op_name__table[node_input_detail['name']][-1] # last sub node
          prev_node_names.append(prev_node_name)

      concat_node_name = self.onnx_node_name
      concat_node = onnx.helper.make_node(
          'Concat',
          inputs=prev_node_names,
          outputs=[concat_node_name],
          axis= utils.channel_last_2_channel_first_axis_mapping( [self.tflite_concat_parser.Axis()] )[0],
          name=concat_node_name
      )

      # update tables
      self.node_list.append(concat_node)

      activative_op = self.tflite_concat_parser.FusedActivationFunction()
      if activative_op == ActivationFunctionType.RELU6:
          clip_name = 'fused_clip_' + self.onnx_node_name
          clip_node = onnx.helper.make_node('Clip',inputs=[self.onnx_node_name],outputs=[clip_name],min=0.0,max=6.0,name=clip_name)
          out_shape_info = onnx.helper.make_tensor_value_info(
              clip_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(clip_node)

      elif activative_op == ActivationFunctionType.RELU:
          relu_name = 'fused_relu_' + self.onnx_node_name
          relu_node = onnx.helper.make_node("Relu",name=relu_name, inputs=[self.onnx_node_name], outputs=[relu_name])
          out_shape_info = onnx.helper.make_tensor_value_info(
              relu_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(relu_node)

      return self.node_list, self.value_infos, self.weight_node_list


class Mul(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

      self.tflite_mul_parser = MulOptions()
      self.tflite_mul_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos) 

  def generate(self, op_name__sub_op_name__table):
      prev_node_names = []
      for input_idx in range(self.op_info.InputsLength()):
          node_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(input_idx))

          prev_node_name = node_input_detail['name']

          if node_input_detail['name'] in op_name__sub_op_name__table:
              prev_node_name = op_name__sub_op_name__table[node_input_detail['name']][-1] # last sub node

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

      mul_node_name = self.onnx_node_name
      mul_node = onnx.helper.make_node(
          'Mul',
          inputs=prev_node_names,
          outputs=[mul_node_name],
          name=mul_node_name
      )

      # update tables
      self.node_list.append(mul_node)

      activative_op = self.tflite_mul_parser.FusedActivationFunction()
      if activative_op == ActivationFunctionType.RELU6:
          clip_name = 'fused_clip_' + self.onnx_node_name
          clip_node = onnx.helper.make_node('Clip',inputs=[self.onnx_node_name],outputs=[clip_name],min=0.0,max=6.0,name=clip_name)
          out_shape_info = onnx.helper.make_tensor_value_info(
              clip_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(clip_node)

      elif activative_op == ActivationFunctionType.RELU:
          relu_name = 'fused_relu_' + self.onnx_node_name
          relu_node = onnx.helper.make_node("Relu",name=relu_name, inputs=[self.onnx_node_name], outputs=[relu_name])
          out_shape_info = onnx.helper.make_tensor_value_info(
              relu_name,
              TensorProto.FLOAT,
              utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(relu_node)

      return self.node_list, self.value_infos, self.weight_node_list