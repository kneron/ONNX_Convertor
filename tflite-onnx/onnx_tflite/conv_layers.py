"""Converters for convolution layers in TFlite
"""
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
import utils
import warnings

from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.TransposeConvOptions import TransposeConvOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding

class Convolution(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

      self.tflite_conv_parser = Conv2DOptions()
      self.tflite_conv_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos)   

  def generate(self):

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(2))

      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])
      bias_array = self.tflite_interpreter.get_tensor(bias_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]

      strides_len = [self.tflite_conv_parser.StrideW(),self.tflite_conv_parser.StrideH()]
      dilation_factor = [self.tflite_conv_parser.DilationWFactor(), self.tflite_conv_parser.DilationHFactor()]

      padding_stradegy = 'NONE' 
      if self.tflite_conv_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME' 
      elif self.tflite_conv_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID' 

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
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
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


      activative_op = self.tflite_conv_parser.FusedActivationFunction()
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


class DepthwiseConvolution(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)
      
      self.tflite_conv_parser = DepthwiseConv2DOptions()
      self.tflite_conv_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos) 

  def generate(self):

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(2))

      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])
      bias_array = self.tflite_interpreter.get_tensor(bias_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]
      channel = weights_array.shape[3]

      strides_len = [self.tflite_conv_parser.StrideW(),self.tflite_conv_parser.StrideH()]
      dilation_factor = [self.tflite_conv_parser.DilationWFactor(),self.tflite_conv_parser.DilationHFactor()]

      padding_stradegy = 'NONE' 
      if self.tflite_conv_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME' 
      elif self.tflite_conv_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID' 

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
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
          name=self.onnx_node_name,

          # goup conv as depthwise conv
          group=int(channel/self.tflite_conv_parser.DepthMultiplier())
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

      activative_op = self.tflite_conv_parser.FusedActivationFunction()
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


class ResizeNearestNeighbor(Layer):

    def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        if utils.ONNX_VERSION_1_4_1 == onnx.__version__:
            warnings.warn(self.__class__.__name__ + ' is implemented by `Upsample` op, and not support `align_corners`,'
                                                    '`half_pixel_centers` attributes.',
                          UserWarning)

            # create constant node
            tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))

            source_width, source_height = self.node_input_detail['shape'].tolist()[1:3]
            target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

            source_size = np.array([1.0, 1.0, source_width, source_height], dtype=np.int32)
            target_siz = np.array([1.0, 1.0, target_width, targwt_height], dtype=np.int32)

            constant_val = target_siz/source_size
            constant_node_name = self.onnx_node_name + '_scales'

            constant_tensor = onnx.helper.make_tensor(
                name=tensor_input_detail['name'],
                data_type=TensorProto.FLOAT,
                dims=constant_val.shape,
                vals=constant_val.ravel())

            constant_node = onnx.helper.make_node(
                op_type="Constant",
                inputs=[],
                outputs=[constant_node_name],
                name=constant_node_name,
                value=constant_tensor)

            constant_info = onnx.helper.make_tensor_value_info(
                name=constant_node_name,
                elem_type=TensorProto.FLOAT,
                shape=constant_val.shape)

            # self.weight_node_list.append(constant_tensor)
            self.node_list.append(constant_node)
            self.value_infos.append(constant_info)

            self.previous_onnx_node_names.extend([constant_node_name])
            resize_nearest_neighbor_node = onnx.helper.make_node(
                op_type='Upsample',
                inputs=self.previous_onnx_node_names,
                outputs=[self.onnx_node_name],
                name=self.onnx_node_name,
                mode='nearest'
            )

            resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
                name=self.onnx_node_name,
                elem_type=TensorProto.FLOAT,
                shape=utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
            )

            # update tables
            self.node_list.append(resize_nearest_neighbor_node)
            self.value_infos.append(resize_nearest_neighbor_info)
        else:
            NotImplementedError('Partially Support ONNX ' + utils.ONNX_VERSION_1_4_1)

        return self.node_list, self.value_infos, self.weight_node_list

class ResizeBilinear(Layer):

    def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

    def generate(self):
        if utils.ONNX_VERSION_1_4_1 == onnx.__version__:
            warnings.warn(self.__class__.__name__ + ' is implemented by `Upsample` op, and not support `align_corners`,'
                                                    '`half_pixel_centers` attributes.',
                          UserWarning)

            # create constant node
            tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))

            source_width, source_height = self.node_input_detail['shape'].tolist()[1:3]
            target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

            source_size = np.array([1.0, 1.0, source_width, source_height], dtype=np.int32)
            target_siz = np.array([1.0, 1.0, target_width, targwt_height], dtype=np.int32)

            constant_val = target_siz/source_size
            constant_node_name = self.onnx_node_name + '_scales'

            constant_tensor = onnx.helper.make_tensor(
                name=tensor_input_detail['name'],
                data_type=TensorProto.FLOAT,
                dims=constant_val.shape,
                vals=constant_val.ravel())

            constant_node = onnx.helper.make_node(
                op_type="Constant",
                inputs=[],
                outputs=[constant_node_name],
                name=constant_node_name,
                value=constant_tensor)

            constant_info = onnx.helper.make_tensor_value_info(
                name=constant_node_name,
                elem_type=TensorProto.FLOAT,
                shape=constant_val.shape)

            # self.weight_node_list.append(constant_tensor)
            self.node_list.append(constant_node)
            self.value_infos.append(constant_info)

            self.previous_onnx_node_names.extend([constant_node_name])
            resize_nearest_neighbor_node = onnx.helper.make_node(
                op_type='Upsample',
                inputs=self.previous_onnx_node_names,
                outputs=[self.onnx_node_name],
                name=self.onnx_node_name,
                mode='linear'
            )

            resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
                name=self.onnx_node_name,
                elem_type=TensorProto.FLOAT,
                shape=utils.tflite2onnx_shape_map(self.node_output_detail['shape'].tolist())
            )

            # update tables
            self.node_list.append(resize_nearest_neighbor_node)
            self.value_infos.append(resize_nearest_neighbor_info)
        else:
            NotImplementedError('Partially Support ONNX ' + utils.ONNX_VERSION_1_4_1)

        return self.node_list, self.value_infos, self.weight_node_list

class TransposeConvolution(Layer):

  def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
      Layer.__init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter)

      self.tflite_tconv_parser = TransposeConvOptions()
      self.tflite_tconv_parser.Init(op_info.BuiltinOptions().Bytes, op_info.BuiltinOptions().Pos)   

  def generate(self):

      self.node_output_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(0))
      self.node_input_detail = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(2))

      output_shape_value = self.tflite_interpreter.get_tensor(self.node_output_detail['index'])

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op_info.Inputs(1))
      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]

      strides_len = [self.tflite_tconv_parser.StrideW(),self.tflite_tconv_parser.StrideH()]

      padding_stradegy = 'NONE' 
      if self.tflite_tconv_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME' 
      elif self.tflite_tconv_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID' 

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

      # make conv onnx node
      self.previous_onnx_node_names.extend([weight_onnx_node_name])
      tconv_onnx_node = onnx.helper.make_node(
          'ConvTranspose',
          inputs= self.previous_onnx_node_names,
          outputs=[self.onnx_node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,

          # TODO: calculate padding for tanspose conv
          #pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          name=self.onnx_node_name,
          group=1
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.onnx_node_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(output_shape_value.tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.node_list.append(tconv_onnx_node)


      return self.node_list, self.value_infos, self.weight_node_list