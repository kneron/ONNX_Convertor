"""Converters for core layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto

import numpy as np

#from . import helper
from base_layer import Layer
#from .exceptions import FeatureNotImplemented, OnnxNotSupport
import utils

class MaxPooling2D(Layer):
  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      kernel_shape = [self.op_info['builtin_options']['filter_width'],self.op_info['builtin_options']['filter_height']]
      strides_len = [self.op_info['builtin_options']['stride_w'],self.op_info['builtin_options']['stride_h']]
      padding_stradegy = self.op_info['builtin_options']['padding']

      input_feature_map_shape = self.node_input_detail['shape']

      max_pool_name = self.onnx_node_name
      max_pool_node = helper.make_node(
          'MaxPool',
          inputs=self.previous_onnx_node_names,
          outputs=[max_pool_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          name=max_pool_name
      )
      out_shape_info = helper.make_tensor_value_info(
          max_pool_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
      )

      # update tables
      self.value_infos.append(out_shape_info)
      self.node_list.append(max_pool_node)

      if 'fused_activation_function' in self.op_info['builtin_options']:
          activative_op = self.op_info['builtin_options']['fused_activation_function']
          if activative_op == 'RELU6':
              clip_name = 'fused_clip_' + self.onnx_node_name
              clip_node = onnx.helper.make_node(
                  'Clip',
                  inputs=[max_pool_name],
                  outputs=[clip_name],
                  min=0.0,
                  max=6.0,
                  name=clip_name
              )
              out_shape_info = helper.make_tensor_value_info(
                  clip_name,
                  TensorProto.FLOAT,
                  utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
              )

              # update tables
              self.value_infos.append(out_shape_info)
              self.node_list.append(clip_node)

          elif activative_op == 'RELU':
              relu_name = 'fused_relu_' + self.onnx_node_name
              relu_node = helper.make_node(
                  "Relu",
                  name=relu_name, 
                  inputs=[max_pool_name], 
                  outputs=[relu_name]
              )
              out_shape_info = helper.make_tensor_value_info(
                  relu_name,
                  TensorProto.FLOAT,
                  utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
              )

              # update tables
              self.value_infos.append(out_shape_info)
              self.node_list.append(relu_node)

      return self.node_list, self.value_infos, self.weight_node_list


class AveragePooling2D(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      kernel_shape = [self.op_info['builtin_options']['filter_width'],self.op_info['builtin_options']['filter_height']]
      strides_len = [self.op_info['builtin_options']['stride_w'],self.op_info['builtin_options']['stride_h']]
      padding_stradegy = self.op_info['builtin_options']['padding']

      input_feature_map_shape = self.node_input_detail['shape']

      avg_pool_name = self.onnx_node_name
      avg_pool_node = helper.make_node(
          'AveragePool',
          inputs=self.previous_onnx_node_names,
          outputs=[avg_pool_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          name=avg_pool_name
      )
      out_shape_info = helper.make_tensor_value_info(
          avg_pool_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
      )

      # update tables
      self.value_infos.append(out_shape_info)
      self.node_list.append(avg_pool_node)

      if 'fused_activation_function' in self.op_info['builtin_options']:
          activative_op = self.op_info['builtin_options']['fused_activation_function']
          if activative_op == 'RELU6':
              clip_name = 'fused_clip_' + self.onnx_node_name
              clip_node = onnx.helper.make_node(
                  'Clip',
                  inputs=[avg_pool_name],
                  outputs=[clip_name],
                  min=0.0,
                  max=6.0,
                  name=clip_name
              )
              out_shape_info = helper.make_tensor_value_info(
                  clip_name,
                  TensorProto.FLOAT,
                  utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
              )

              # update tables
              self.value_infos.append(out_shape_info)
              self.node_list.append(clip_node)

          elif activative_op == 'RELU':
              relu_name = 'fused_relu_' + self.onnx_node_name
              relu_node = helper.make_node(
                  "Relu",
                  name=relu_name, 
                  inputs=[avg_pool_name], 
                  outputs=[relu_name]
              )
              out_shape_info = helper.make_tensor_value_info(
                  relu_name,
                  TensorProto.FLOAT,
                  utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
              )

              # update tables
              self.value_infos.append(out_shape_info)
              self.node_list.append(relu_node)

      return self.node_list, self.value_infos, self.weight_node_list


class Mean(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):
      flag_keep_dims = self.op_info['builtin_options']['keep_dims']

      mean_node_name = self.onnx_node_name
      mean_node = onnx.helper.make_node(
          'GlobalAveragePool',
          inputs=self.previous_onnx_node_names,
          outputs=[mean_node_name]
      )

      # update tables
      self.node_list.append(mean_node)


      ##################  add squeeze  ###############
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############

      if not flag_keep_dims:
          squeeze_node_name = 'squeeze_node_after_gap_' + self.onnx_node_name
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
              self.node_output_detail['shape'].tolist()
          )

          # update tables
          self.value_infos.append(out_shape_info)
          self.node_list.append(squeeze_node)
      
      
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############
      ##################  add squeeze  ###############

      return self.node_list, self.value_infos, self.weight_node_list


