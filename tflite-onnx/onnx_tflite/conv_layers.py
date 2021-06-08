"""Converters for convolution layers in TFlite
"""
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
from aact_layers import defused_activation_node_generator
import tflite_utils
import logging

from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.TransposeConvOptions import TransposeConvOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding
from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
from tflite.ResizeBilinearOptions import ResizeBilinearOptions

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

      strides_len = [self.tflite_conv_parser.StrideH(),self.tflite_conv_parser.StrideW()]
      dilation_factor = [self.tflite_conv_parser.DilationHFactor(), self.tflite_conv_parser.DilationWFactor()]

      #Generate Quantization Info and Reverse Quantization for Weights and Bias
      output_quantization_info = node_output_detail.get("quantization_parameters", {})
      output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
      input_quantization_info = node_input_detail.get("quantization_parameters", {})
      input_quantization_info["dtype"] = str(node_input_detail["dtype"]).split(".")[1].split("'")[0]
      weight_quantization_info = weights_node_info.get("quantization_parameters", {})
      weight_quantization_info["dtype"] = str(weights_node_info["dtype"]).split(".")[1].split("'")[0]
      bias_quantization_info = bias_node_info.get("quantization_parameters", {})
      bias_quantization_info["dtype"] = str(bias_node_info["dtype"]).split(".")[1].split("'")[0]

    #   input_quantization_info_clean = utils.get_quantization_info_in_array(input_quantization_info)
    #   output_quantization_info_clean = utils.get_quantization_info_in_array(output_quantization_info)
    #   weight_quantization_info_clean = utils.get_quantization_info_in_array(weight_quantization_info)
    #   bias_quantization_info_clean = utils.get_quantization_info_in_array(bias_quantization_info)
      #Nested weight and bias into input
      input_quantization_info["weight"] =  weight_quantization_info
      input_quantization_info["bias"] = bias_quantization_info

      weights_array = np.array(weights_array, dtype = np.dtype("f4"))
      if weight_quantization_info["scales"]:
          weights_array = (weights_array - weight_quantization_info["zero_points"][0]) * weight_quantization_info["scales"][0]
      bias_array = np.array(bias_array, dtype = np.dtype("f4"))
      if bias_quantization_info["scales"]:
          bias_array = (bias_array - bias_quantization_info["zero_points"][0]) * bias_quantization_info["scales"][0]

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
          pads=tflite_utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
          name=self.node_name,
          group=1
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          tflite_utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
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
      
      quantization_info = {}
      quantization_info[weight_onnx_node_name] = weight_quantization_info
      quantization_info[bias_onnx_node_name] = bias_quantization_info
      quantization_info[previous_onnx_node_names[0]] = input_quantization_info
      quantization_info[self.node_name] = output_quantization_info

      return self.node_list, self.value_infos, self.weight_node_list, quantization_info

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
      
      #Generate Quantization Info and Reverse Quantization for Weights and Bias
      output_quantization_info = node_output_detail.get("quantization_parameters", {})
      output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
      input_quantization_info = node_input_detail.get("quantization_parameters", {})
      input_quantization_info["dtype"] = str(node_input_detail["dtype"]).split(".")[1].split("'")[0]
      weight_quantization_info = weights_node_info.get("quantization_parameters", {})
      weight_quantization_info["dtype"] = str(weights_node_info["dtype"]).split(".")[1].split("'")[0]
      bias_quantization_info = bias_node_info.get("quantization_parameters", {})
      bias_quantization_info["dtype"] = str(bias_node_info["dtype"]).split(".")[1].split("'")[0]
      #Nested weight and bias into input
      input_quantization_info["weight"] =  weight_quantization_info
      input_quantization_info["bias"] = bias_quantization_info

      weights_array = np.array(weights_array, dtype = np.dtype("f4"))
      if weight_quantization_info["scales"]:
          weights_array = (weights_array - weight_quantization_info["zero_points"][0]) * weight_quantization_info["scales"][0]
      bias_array = np.array(bias_array, dtype = np.dtype("f4"))
      if bias_quantization_info["scales"]:
          bias_array = (bias_array - bias_quantization_info["zero_points"][0]) * bias_quantization_info["scales"][0]

      kernel_shape=[weights_array.shape[1], weights_array.shape[2]]
      channel = weights_array.shape[3]

      strides_len = [self.tflite_conv_parser.StrideH(),self.tflite_conv_parser.StrideW()]
      dilation_factor = [self.tflite_conv_parser.DilationHFactor(),self.tflite_conv_parser.DilationWFactor()]

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
          pads=tflite_utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, dilation_factor, padding_stradegy),
          dilations=dilation_factor,
          name=self.node_name,

          # goup conv as depthwise conv
          group=int(channel/self.tflite_conv_parser.DepthMultiplier())
      )

       # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          tflite_utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
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
      
      quantization_info = {}
      quantization_info[weight_onnx_node_name] = weight_quantization_info
      quantization_info[bias_onnx_node_name] = bias_quantization_info
      quantization_info[previous_onnx_node_names[0]] = input_quantization_info
      quantization_info[self.node_name] = output_quantization_info

      return self.node_list, self.value_infos, self.weight_node_list, quantization_info
      
  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_conv_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)


class ResizeNearestNeighbor(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

        self.tflite_resize_nn_parser = ResizeNearestNeighborOptions()
        self.tflite_resize_nn_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

    def generate(self):
        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

        # create scale constant node
        tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))

        source_width, source_height = node_input_detail['shape'].tolist()[1:3]
        target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

        source_size = np.array([1.0, 1.0, source_height, source_width], dtype=np.float)
        target_siz = np.array([1.0, 1.0, targwt_height, target_width], dtype=np.float)

        scale_val = target_siz/source_size
        scale_constant_node = tflite_utils.create_constant_node(self.node_name + '_scales' ,scale_val.shape ,scale_val)

        constant_info = onnx.helper.make_tensor_value_info(
            name=scale_constant_node.name,
            elem_type=TensorProto.FLOAT,
            shape=scale_val.shape)

        self.node_list.append(scale_constant_node)
        self.value_infos.append(constant_info)

        # create roi constant node
        roi_constant_node = tflite_utils.create_constant_node(self.node_name + 'resize_roi', [], np.array([-1],dtype=np.float32))
        self.node_list.append(roi_constant_node)

        previous_onnx_node_names = self.input_nodes_name.copy()
        previous_onnx_node_names.extend([roi_constant_node.name, scale_constant_node.name])
        resize_nearest_neighbor_node = onnx.helper.make_node(
            op_type='Resize',
            inputs=previous_onnx_node_names,
            outputs=[self.node_name],
            name=self.node_name,
            mode='nearest',
            coordinate_transformation_mode = 'align_corners' if self.tflite_resize_nn_parser.AlignCorners() == True else 'half_pixel'
        )

        resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
            name=self.node_name,
            elem_type=TensorProto.FLOAT,
            shape=tflite_utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        # update tables
        self.node_list.append(resize_nearest_neighbor_node)
        self.value_infos.append(resize_nearest_neighbor_info)

        quantization_info = {}
        return self.node_list, self.value_infos, self.weight_node_list, quantization_info


class ResizeBilinear(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

        self.tflite_resize_bilinear_parser = ResizeBilinearOptions()
        self.tflite_resize_bilinear_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

    def generate(self):
        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

        # create scale constant node
        tensor_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))

        source_width, source_height = node_input_detail['shape'].tolist()[1:3]
        target_width, targwt_height = self.tflite_interpreter.get_tensor(tensor_input_detail['index']).tolist()

        source_size = np.array([1.0, 1.0, source_height, source_width], dtype=np.float)
        target_siz = np.array([1.0, 1.0, targwt_height, target_width], dtype=np.float)

        scale_val = target_siz/source_size
        scale_constant_node = tflite_utils.create_constant_node(self.node_name + '_scales' ,scale_val.shape ,scale_val)

        constant_info = onnx.helper.make_tensor_value_info(
            name=scale_constant_node.name,
            elem_type=TensorProto.FLOAT,
            shape=scale_val.shape)

        self.node_list.append(scale_constant_node)
        self.value_infos.append(constant_info)

        # create roi constant node
        roi_constant_node = tflite_utils.create_constant_node(self.node_name + 'resize_roi', [], np.array([-1],dtype=np.float32))
        self.node_list.append(roi_constant_node)

        previous_onnx_node_names = self.input_nodes_name.copy()
        previous_onnx_node_names.extend([roi_constant_node.name, scale_constant_node.name])
        resize_nearest_neighbor_node = onnx.helper.make_node(
            op_type='Resize',
            inputs=previous_onnx_node_names,
            outputs=[self.node_name],
            name=self.node_name,
            mode='linear',
            coordinate_transformation_mode = 'align_corners' if self.tflite_resize_bilinear_parser.AlignCorners() == True else 'half_pixel'
        )

        resize_nearest_neighbor_info = onnx.helper.make_tensor_value_info(
            name=self.node_name,
            elem_type=TensorProto.FLOAT,
            shape=tflite_utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        # update tables
        self.node_list.append(resize_nearest_neighbor_node)
        self.value_infos.append(resize_nearest_neighbor_info)

        quantization_info = {}
        return self.node_list, self.value_infos, self.weight_node_list, quantization_info


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

      strides_len = [self.tflite_tconv_parser.StrideH(),self.tflite_tconv_parser.StrideW()]

      #Generate Quantization Info and Reverse Quantization for Weights and Bias
      output_quantization_info = node_output_detail.get("quantization_parameters", {})
      output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
      input_quantization_info = node_input_detail.get("quantization_parameters", {})
      input_quantization_info["dtype"] = str(node_input_detail["dtype"]).split(".")[1].split("'")[0]
      weight_quantization_info = weights_node_info.get("quantization_parameters", {})
      weight_quantization_info["dtype"] = str(weights_node_info["dtype"]).split(".")[1].split("'")[0]
      weights_array = np.array(weights_array, dtype = np.dtype("f4"))
      if weight_quantization_info["scales"]:
          weights_array = (weights_array - weight_quantization_info["zero_points"][0]) * weight_quantization_info["scales"][0]
      #Nested weight quantization info into input
      input_quantization_info["weight"] = weight_quantization_info

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
          #pads = tflite_utils.getPadding(input_feature_map_shape, kernel_shape, strides_len, padding_stradegy),
          name=self.node_name,
          group=1
      )

      # original layer output
      out_shape_info = onnx.helper.make_tensor_value_info(
          self.node_name,
          TensorProto.FLOAT,
          tflite_utils.tflite2onnx_shape_map(output_shape_value.tolist())
      )
      self.value_infos.append(out_shape_info)

      # add weight, bias node
      self.weight_node_list.append(weight_onnx_node)
      self.node_list.append(tconv_onnx_node)

      quantization_info = {}
      quantization_info[weight_onnx_node_name] = weight_quantization_info
      quantization_info[previous_onnx_node_names[0]] = input_quantization_info
      quantization_info[self.node_name] = output_quantization_info

      return self.node_list, self.value_infos, self.weight_node_list, quantization_info