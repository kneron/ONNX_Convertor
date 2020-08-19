"""Converters for convolution layers in TFlite
"""
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
from aact_layers import defused_activation_node_generator
import utils
import warnings

from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.TransposeConvOptions import TransposeConvOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding

class Convolution(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_conv_parser = Conv2DOptions()
      self.tflite_conv_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(2))

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

      input_feature_map_shape = node_input_detail['shape']

      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (0, 3, 1, 2))

      # make weight onnx node
      weight_onnx_node_name = self.node_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make bias onnx node
      bias_onnx_node_name = self.node_name + "_bias"
      bias_onnx_node = onnx.helper.make_tensor(
          bias_onnx_node_name,
          TensorProto.FLOAT,
          bias_array.shape,
          bias_array.flatten().tolist()
      )

      # make conv onnx node
      previous_onnx_node_names = self.input_nodes_name.copy()
      previous_onnx_node_names.extend([weight_onnx_node_name, bias_onnx_node_name])
      conv_onnx_node = onnx.helper.make_node(
          'Conv',
          inputs= previous_onnx_node_names,
          outputs=[self.node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
          name=self.node_name,
          group=1
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.weight_node_list.append(bias_onnx_node)
      self.node_list.append(conv_onnx_node)

      # change output node's input_name
      for o_n in self.output_nodes:
          for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
              if o_n_i_n == self.node_name:
                  o_n.input_nodes_name[idx] = self.node_list[-1].name

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_conv_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)


class DepthwiseConvolution(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_conv_parser = DepthwiseConv2DOptions()
      self.tflite_conv_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(2))

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

      input_feature_map_shape = node_input_detail['shape']


      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (3, 0, 1, 2))

      # make weight onnx node
      weight_onnx_node_name = self.node_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make bias onnx node
      bias_onnx_node_name = self.node_name + "_bias"
      bias_onnx_node = onnx.helper.make_tensor(
          bias_onnx_node_name,
          TensorProto.FLOAT,
          bias_array.shape,
          bias_array.flatten().tolist()
      )

      # make conv onnx node
      previous_onnx_node_names = self.input_nodes_name.copy()
      previous_onnx_node_names.extend([weight_onnx_node_name, bias_onnx_node_name])
      conv_onnx_node = onnx.helper.make_node(
          'Conv',
          inputs= previous_onnx_node_names,
          outputs=[self.node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,
          pads=utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
          name=self.node_name,

          # goup conv as depthwise conv
          group=int(channel/self.tflite_conv_parser.DepthMultiplier())
      )

       # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
      )

      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.weight_node_list.append(bias_onnx_node)
      self.node_list.append(conv_onnx_node)

      # change output node's input_name
      for o_n in self.output_nodes:
          for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
              if o_n_i_n == self.node_name:
                  o_n.input_nodes_name[idx] = self.node_list[-1].name

      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_conv_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)


class ResizeNearestNeighbor(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        if utils.ONNX_VERSION_1_4_1 == onnx.__version__:
            warnings.warn(self.__class__.__name__ + ' is implemented by `Upsample` op, and not support `align_corners`,'
                                                    '`half_pixel_centers` attributes.',
                          UserWarning)

            node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
            node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

            # create constant node
            tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))

            source_width, source_height = node_input_detail['shape'].tolist()[1:3]
            target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

            source_size = np.array([1.0, 1.0, source_width, source_height], dtype=np.int32)
            target_siz = np.array([1.0, 1.0, target_width, targwt_height], dtype=np.int32)

            constant_val = target_siz/source_size
            constant_node_name = self.node_name + '_scales'

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

            previous_onnx_node_names = self.input_nodes_name.copy()
            previous_onnx_node_names.extend([constant_node_name])
            resize_nearest_neighbor_node = onnx.helper.make_node(
                op_type='Upsample',
                inputs=previous_onnx_node_names,
                outputs=[self.node_name],
                name=self.node_name,
                mode='nearest'
            )

            resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
                name=self.node_name,
                elem_type=TensorProto.FLOAT,
                shape=utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
            )

            # update tables
            self.node_list.append(resize_nearest_neighbor_node)
            self.value_infos.append(resize_nearest_neighbor_info)
        else:
            NotImplementedError('Partially Support ONNX ' + utils.ONNX_VERSION_1_4_1)

        return self.node_list, self.value_infos, self.weight_node_list


class ResizeBilinear(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        if utils.ONNX_VERSION_1_4_1 == onnx.__version__:
            warnings.warn(self.__class__.__name__ + ' is implemented by `Upsample` op, and not support `align_corners`,'
                                                    '`half_pixel_centers` attributes.',
                          UserWarning)

            node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
            node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

            # create constant node
            tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))

            source_width, source_height = node_input_detail['shape'].tolist()[1:3]
            target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

            source_size = np.array([1.0, 1.0, source_width, source_height], dtype=np.int32)
            target_siz = np.array([1.0, 1.0, target_width, targwt_height], dtype=np.int32)

            constant_val = target_siz/source_size
            constant_node_name = self.node_name + '_scales'

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

            previous_onnx_node_names = self.input_nodes_name.copy()
            previous_onnx_node_names.extend([constant_node_name])
            resize_nearest_neighbor_node = onnx.helper.make_node(
                op_type='Upsample',
                inputs=previous_onnx_node_names,
                outputs=[self.node_name],
                name=self.node_name,
                mode='linear'
            )

            resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
                name=self.node_name,
                elem_type=TensorProto.FLOAT,
                shape=utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
            )

            # update tables
            self.node_list.append(resize_nearest_neighbor_node)
            self.value_infos.append(resize_nearest_neighbor_info)
        else:
            NotImplementedError('Partially Support ONNX ' + utils.ONNX_VERSION_1_4_1)

        return self.node_list, self.value_infos, self.weight_node_list


class TransposeConvolution(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_tconv_parser = TransposeConvOptions()
      self.tflite_tconv_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):

      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(2))

      output_shape_value = self.tflite_interpreter.get_tensor(node_output_detail['index'])

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]

      strides_len = [self.tflite_tconv_parser.StrideW(),self.tflite_tconv_parser.StrideH()]

      padding_stradegy = 'NONE'
      if self.tflite_tconv_parser.Padding() is Padding.SAME:
          padding_stradegy = 'SAME'
      elif self.tflite_tconv_parser.Padding() is Padding.VALID:
          padding_stradegy = 'VALID'

      input_feature_map_shape = node_input_detail['shape']

      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (3, 0, 1, 2))

      # make weight onnx node
      weight_onnx_node_name = self.node_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make conv onnx node
      previous_onnx_node_names = self.input_nodes_name.copy()
      previous_onnx_node_names.extend([weight_onnx_node_name])
      tconv_onnx_node = onnx.helper.make_node(
          'ConvTranspose',
          inputs= previous_onnx_node_names,
          outputs=[self.node_name],
          kernel_shape=kernel_shape,
          strides=strides_len,

          # TODO: calculate padding for tanspose conv
          #pads = utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          name=self.node_name,
          group=1
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          utils.tflite2onnx_shape_map(output_shape_value.tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.node_list.append(tconv_onnx_node)


      return self.node_list, self.value_infos, self.weight_node_list