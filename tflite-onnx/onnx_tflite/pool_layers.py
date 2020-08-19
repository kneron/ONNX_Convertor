"""Converters for core layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
from aact_layers import defused_activation_node_generator
import utils

from tflite.ReducerOptions import ReducerOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.Padding import Padding
from tflite.ActivationFunctionType import ActivationFunctionType

class MaxPooling2D(Layer):
  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_maxpool_parser = Pool2DOptions()
      self.tflite_maxpool_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):

      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      kernel_shape = [self.tflite_maxpool_parser.FilterWidth(),self.tflite_maxpool_parser.FilterHeight()]
      strides_len = [self.tflite_maxpool_parser.StrideW(),self.tflite_maxpool_parser.StrideH()]

      padding_stradegy = 'NONE' 
      if self.tflite_maxpool_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME' 
      elif self.tflite_maxpool_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID' 

      input_feature_map_shape = node_input_detail['shape']

      max_pool_name = self.node_name
      max_pool_node = helper.make_node(
          'MaxPool',
          inputs=self.input_nodes_name,
          outputs=[max_pool_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, None, padding_stradegy),
          name=max_pool_name
      )
      out_shape_info = helper.make_tensor_value_info(
          max_pool_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
      )

      # update tables
      self.value_infos.append(out_shape_info)
      self.node_list.append(max_pool_node)

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_maxpool_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)

class AveragePooling2D(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_avgpool_parser = Pool2DOptions()
      self.tflite_avgpool_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):

      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      kernel_shape = [self.tflite_avgpool_parser.FilterWidth(),self.tflite_avgpool_parser.FilterHeight()]
      strides_len = [self.tflite_avgpool_parser.StrideW(),self.tflite_avgpool_parser.StrideH()]

      padding_stradegy = 'NONE' 
      if self.tflite_avgpool_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME' 
      elif self.tflite_avgpool_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID' 

      input_feature_map_shape = node_input_detail['shape']

      avg_pool_name = self.node_name
      avg_pool_node = helper.make_node(
          'AveragePool',
          inputs=self.input_nodes_name,
          outputs=[avg_pool_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, None, padding_stradegy),
          name=avg_pool_name
      )
      out_shape_info = helper.make_tensor_value_info(
          avg_pool_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
      )

      # update tables
      self.value_infos.append(out_shape_info)
      self.node_list.append(avg_pool_node)

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_avgpool_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)

class Mean(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_mean_parser = ReducerOptions()
      self.tflite_mean_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))

      flag_keep_dims = self.tflite_mean_parser.KeepDims()

      mean_node_name = self.node_name
      mean_node = onnx.helper.make_node(
          'GlobalAveragePool',
          inputs=self.input_nodes_name,
          outputs=[mean_node_name],
          name=mean_node_name
      )

      # update tables
      self.node_list.append(mean_node)


      ##################  add squeeze  ###############
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############

      if not flag_keep_dims:
          squeeze_node_name = 'squeeze_node_after_gap_' + self.node_name
          squeeze_node = onnx.helper.make_node(
              'Squeeze',
              inputs=[mean_node_name],
              outputs=[squeeze_node_name],
              axes= [2,3],
              name=squeeze_node_name
          )
          out_shape_info = helper.make_tensor_value_info(
              squeeze_node_name,
              TensorProto.FLOAT,
              node_output_detail['shape'].tolist()
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(squeeze_node)

          # change output node's input_name
          for o_n in self.output_nodes:
             for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
                 if o_n_i_n == self.node_name:
                    o_n.input_nodes_name[idx] = squeeze_node_name
      
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############



      return self.node_list, self.value_infos, self.weight_node_list


