"""Converters for core layers in Keras
"""
import onnx 
from onnx import helper
from onnx import AttributeProto, TensorProto
import numpy as np
from base_layer import Layer
from aact_layers import defused_activation_node_generator
import utils

from tflite.ReshapeOptions import ReshapeOptions
from tflite.L2NormOptions import L2NormOptions
from tflite.SqueezeOptions import SqueezeOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.ActivationFunctionType import ActivationFunctionType

class Dense(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_fc_parser = FullyConnectedOptions()
      self.tflite_fc_parser.Init(self.op.BuiltinOptions().Bytes, self.op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      # if previous shape is looks like [1,1,1,x], we can do squeeze
      previous_shape = node_input_detail['shape']
      squeeze_node_name = 'squeeze_node_before_gemm_' + self.node_name
      if previous_shape.size == 4:
        if previous_shape[0] == 1 and previous_shape[1] == 1 and previous_shape[2] == 1: 

            ##################  add squeeze  ###############
            squeeze_node = onnx.helper.make_node(
                'Squeeze',
                inputs= self.input_nodes_name,
                outputs=[squeeze_node_name],
                axes= [2,3],
                name=squeeze_node_name
            )

            self.node_list.append(squeeze_node)

            # change squeeze to new input for gemm
            self.input_nodes_name = [squeeze_node_name]      

      fc_name = self.node_name

      weights_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
      bias_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(2))

      weights_array = self.tflite_interpreter.get_tensor(weights_node_info['index'])
      bias_array = self.tflite_interpreter.get_tensor(bias_node_info['index'])

      # transpose because shape define diffent between tflite and onnx
      weights_array = np.transpose(weights_array, (1,0))


      # make weight onnx node
      weight_onnx_node_name = fc_name + "_weight"
      weight_onnx_node = onnx.helper.make_tensor(
          weight_onnx_node_name,
          TensorProto.FLOAT,
          weights_array.shape,
          weights_array.flatten().tolist()
      )

      # make bias onnx node
      bias_onnx_node_name = fc_name + "_bias"
      bias_onnx_node = onnx.helper.make_tensor(
          bias_onnx_node_name,
          TensorProto.FLOAT,
          bias_array.shape,
          bias_array.flatten().tolist()
      )

      # make FC onnx node
      node_name_before_fc = []
      node_name_before_fc.extend(self.input_nodes_name)
      node_name_before_fc.append(weight_onnx_node_name)
      node_name_before_fc.append(bias_onnx_node_name)
      fc_onnx_node = helper.make_node(
          op_type   = 'Gemm',
          inputs    = node_name_before_fc,
          outputs   = [fc_name],
          name      = fc_name,
          alpha     = 1.0,
          beta      = 1.0,
          transA    = 0,
          transB    = 0
      )

      out_shape_info = helper.make_tensor_value_info(
          fc_name,
          TensorProto.FLOAT,
          node_output_detail['shape'].tolist()
      )

      # update tables
      self.value_infos.append(out_shape_info)
      self.weight_node_list.append(weight_onnx_node)
      self.weight_node_list.append(bias_onnx_node)
      self.node_list.append(fc_onnx_node)

      # change output node's input_name
      for o_n in self.output_nodes:
          for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
              if o_n_i_n == self.node_name:
                  o_n.input_nodes_name[idx] = self.node_list[-1].name


      return self.node_list, self.value_infos, self.weight_node_list

  def defuse_activation_function(self):
      return defused_activation_node_generator(
          activation_function_type=self.tflite_fc_parser.FusedActivationFunction(),
          op=self.op,
          tflite_interpreter=self.tflite_interpreter)

class Reshape(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_reshape_parser = ReshapeOptions()
      self.tflite_reshape_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):
      node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
      node_input_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(0))

      out_dim = node_output_detail['shape']
      in_dim = node_input_detail['shape']

      dims = list(range(len(in_dim)))
      dims = dims[:1] + dims[2:] + dims[1:2]

      # add transpose
      transpose_before_node_name = 'transpose_node_before_reshape_' + self.node_name
      transpose_before_node = onnx.helper.make_node(
          'Transpose',
          inputs=self.input_nodes_name,
          outputs=[transpose_before_node_name],
          perm=dims,
          name=transpose_before_node_name
      )
      # update tables
      self.node_list.append(transpose_before_node)


      reshape_node_name = self.node_name
      shape_tensor_name = 'shape_tensor_' + self.node_name
      shape_node_name = 'shape_const_' + self.node_name


      # no attribute 'new_shape', should be op 'squeeze'
      if not self.tflite_reshape_parser.NewShapeIsNone:
        new_shape = np.array(self.tflite_reshape_parser.NewShapeAsNumpy(), dtype='int64')
      else:
        new_shape = np.array(out_dim, dtype='int64')

      shape_tensor = onnx.helper.make_tensor(shape_tensor_name,TensorProto.INT64,new_shape.shape, new_shape)
      shape_node = helper.make_node("Constant",[],[shape_node_name],name=shape_node_name,value=shape_tensor)

      reshape_node = onnx.helper.make_node(
          'Reshape',
          inputs=[transpose_before_node_name, shape_node_name],
          outputs=[reshape_node_name],
          name=reshape_node_name
      )
      # update tables
      self.node_list.append(shape_node)
      self.node_list.append(reshape_node)

      # no attribute 'new_shape', 
      if self.tflite_reshape_parser.NewShapeIsNone:
        dims = list(range(len(out_dim)))
        dims = dims[:1] + dims[-1:] + dims[1:-1]
        # add transpose
        transpose_after_node_name = 'transpose_node_after_reshape_' + self.node_name
        transpose_after_node = onnx.helper.make_node(
            'Transpose',
            inputs=[reshape_node_name],
            outputs=[transpose_after_node_name],
            perm=dims,
            name=transpose_after_node_name
        )

        # update tables
        self.node_list.append(transpose_after_node)

        # change output node's input_name
        for o_n in self.output_nodes:
            for idx, o_n_i_n in enumerate(o_n.input_nodes_name):
                if o_n_i_n == self.node_name:
                   o_n.input_nodes_name[idx] = transpose_after_node_name

      return self.node_list, self.value_infos, self.weight_node_list


class Pad(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

  def generate(self):
      # tflite pad :[[0 0]
      #             [2 1]           onnx pad :
      #             [2 1]     ==       [0,0,2,2,3,3,1,1]          
      #             [3 3]]          

      # create constant node
      pad_node_detail = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
      pad_param = self.tflite_interpreter.get_tensor(pad_node_detail['index']).tolist()

      pad_w0 = pad_param[1][0]
      pad_h0 = pad_param[2][0]

      pad_w = pad_param[1][1]
      pad_h = pad_param[2][1]

      pad_ch_w = pad_param[3][0]
      pad_ch_h = pad_param[3][1]

      # build node
      pad_name = self.node_name
      pad_node = helper.make_node(
          'Pad', 
          self.input_nodes_name, 
          [pad_name], 
          mode='constant', 
          value=0.0, 
          pads=[0,0,pad_w0,pad_h0,pad_ch_w,pad_ch_h,pad_w,pad_h], 
          name=pad_name 
      )

      # update tables
      self.node_list.append(pad_node)

      return self.node_list, self.value_infos, self.weight_node_list

class Squeeze(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_squeeze_parser = SqueezeOptions()
      self.tflite_squeeze_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):
      squeeze_node_name = self.node_name
      squeeze_node = onnx.helper.make_node(
          'Squeeze',
          inputs=self.input_nodes_name,
          outputs=[squeeze_node_name],
          axes= utils.channel_last_2_channel_first_axis_mapping( self.tflite_squeeze_parser.SqueezeDimsAsNumpy().tolist() ),
          name=squeeze_node_name
      )

      # update tables
      self.node_list.append(squeeze_node)

      return self.node_list, self.value_infos, self.weight_node_list

class L2Normalization(Layer):

  def __init__(self, op, op_type, tflite_interpreter):
      Layer.__init__(self, op, op_type, tflite_interpreter)

      self.tflite_l2norm_parser = L2NormOptions()
      self.tflite_l2norm_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

  def generate(self):
      l2norm_node_name = self.node_name
      l2norm_node = onnx.helper.make_node(
          'LpNormalization',
          inputs=self.input_nodes_name,
          outputs=[l2norm_node_name],

          # along last dimension
          axis=-1,

          # L2
          p=2,

          name=l2norm_node_name
      )

      # update tables
      self.node_list.append(l2norm_node)

      return self.node_list, self.value_infos, self.weight_node_list