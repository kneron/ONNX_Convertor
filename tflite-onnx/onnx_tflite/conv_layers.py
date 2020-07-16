"""Converters for convolution layers in TFlite
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
import utils

class Convolution(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)


  def generate(self):

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op_info['inputs'][1])
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op_info['inputs'][2])

      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])
      bias_array = self.tflite_interpreter.get_tensor(bias_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]

      strides_len = [self.op_info['builtin_options']['stride_w'],self.op_info['builtin_options']['stride_h']]
      padding_stradegy = self.op_info['builtin_options']['padding']

      input_feature_map_shape = self.node_input_detail['shape']

      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (0, 3, 1, 2))

      # make weight onnx node
      weight_onnx_node_name = self.onnx_node_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make bias onnx node
      bias_onnx_node_name = self.onnx_node_name + "_bias"
      bias_onnx_node = onnx.helper.make_tensor(
          bias_onnx_node_name,
          TensorProto.FLOAT,
          bias_array.shape,
          bias_array.flatten().tolist()
      )

      # make conv onnx node
      self.previous_onnx_node_names.extend([weight_onnx_node_name, bias_onnx_node_name])
      conv_onnx_node = onnx.helper.make_node(
          'Conv',
          inputs= self.previous_onnx_node_names,
          outputs=[self.onnx_node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          dilations=[self.op_info['builtin_options']['dilation_w_factor'],self.op_info['builtin_options']['dilation_h_factor']],
          name=self.onnx_node_name,
          group=1 
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.onnx_node_name, 
          TensorProto.FLOAT, 
          utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.weight_node_list.append(bias_onnx_node)
      self.node_list.append(conv_onnx_node)

      if 'fused_activation_function' in self.op_info['builtin_options']:

          activative_op = self.op_info['builtin_options']['fused_activation_function']
          if activative_op == 'RELU6':
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

          elif activative_op == 'RELU':
              relu_name = 'fused_relu_' + self.onnx_node_name
              relu_node = onnx.helper.make_node("Relu",name=relu_name, inputs=[self.onnx_node_name], outputs=[relu_name])
              out_shape_info = onnx.helper.make_tensor_value_info(
                  relu_name,
                  TensorProto.
                  FLOAT,utils.tflite2onnx_shape_map((self.node_output_detail['shape'].tolist()))
              )

              # update tables
              self.value_infos.append(out_shape_info)
              self.node_list.append(relu_node)


      return self.node_list, self.value_infos, self.weight_node_list


class DepthwiseConvolution(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

  def generate(self):

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op_info['inputs'][1])
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op_info['inputs'][2])

      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])
      bias_array = self.tflite_interpreter.get_tensor(bias_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]
      channel = weights_array.shape[3]

      strides_len = [self.op_info['builtin_options']['stride_w'],self.op_info['builtin_options']['stride_h']]
      padding_stradegy = self.op_info['builtin_options']['padding']
      
      input_feature_map_shape = self.node_input_detail['shape']

      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (3, 0, 1, 2))

      # make weight onnx node
      weight_onnx_node_name = self.onnx_node_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make bias onnx node
      bias_onnx_node_name = self.onnx_node_name + "_bias"
      bias_onnx_node = onnx.helper.make_tensor(
          bias_onnx_node_name,
          TensorProto.FLOAT,
          bias_array.shape,
          bias_array.flatten().tolist()
      )

      # make conv onnx node
      self.previous_onnx_node_names.extend([weight_onnx_node_name, bias_onnx_node_name])
      conv_onnx_node = onnx.helper.make_node(
          'Conv',
          inputs= self.previous_onnx_node_names,
          outputs=[self.onnx_node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          dilations=[self.op_info['builtin_options']['dilation_w_factor'],self.op_info['builtin_options']['dilation_h_factor']],
          name=self.onnx_node_name,

          # goup conv as depthwise conv
          group=channel
      )

       # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.onnx_node_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.weight_node_list.append(bias_onnx_node)
      self.node_list.append(conv_onnx_node)

      if 'fused_activation_function' in self.op_info['builtin_options']:

          activative_op = self.op_info['builtin_options']['fused_activation_function']
          if activative_op == 'RELU6':
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

          elif activative_op == 'RELU':
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


