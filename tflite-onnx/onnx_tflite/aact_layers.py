"""Converters for core layers in TFlite
"""
import abc
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
from tflite.LeakyReluOptions import LeakyReluOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.BuiltinOperator import BuiltinOperator
import numpy as np
from base_layer import Layer
import utils
import warnings


def defused_activation_node_generator(activation_function_type: int, op, tflite_interpreter):

    if ActivationFunctionType.NONE == activation_function_type:
        defused_activation_node = None

    elif ActivationFunctionType.RELU == activation_function_type:
        defused_activation_node = ReluDefused(op=op, op_type=BuiltinOperator.RELU, tflite_interpreter=tflite_interpreter)

    elif ActivationFunctionType.RELU_N1_TO_1 == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('RELU_N1_TO_1'), UserWarning)
        defused_activation_node = None

    elif ActivationFunctionType.RELU6 == activation_function_type:
        defused_activation_node = Relu6Defused(op=op, op_type=BuiltinOperator.RELU6, tflite_interpreter=tflite_interpreter)

    elif ActivationFunctionType.TANH == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('TANH'), UserWarning)
        defused_activation_node = None

    elif ActivationFunctionType.SIGN_BIT == activation_function_type:
        warnings.warn('Not Support {} Fused Activation Currently.'.format('SIGN_BIT'), UserWarning)
        defused_activation_node = None

    else:
        warnings.warn('Fused Activation Type {} Not in Specification.'.format(activation_function_type), UserWarning)
        defused_activation_node = None


    return defused_activation_node


# Defused Activation Layer
class ActivationDefused(Layer, metaclass=abc.ABCMeta):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)
        self.node_name = '{}_Fused'.format(self.node_name)


class ReluDefused(ActivationDefused):

    def __init__(self, op, op_type, tflite_interpreter):
        ActivationDefused.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        relu_name = self.node_name
        relu_node = helper.make_node("Relu",
                                     name=relu_name,
                                     inputs=self.input_nodes_name,
                                     outputs=[relu_name])

        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = onnx.helper.make_tensor_value_info(
            relu_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(relu_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}


class Relu6Defused(ActivationDefused):

    def __init__(self, op, op_type, tflite_interpreter):
        ActivationDefused.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        clip_name = self.node_name

        six = np.array([6.0])
        zero = np.array([0.0])
        # onnx clip only support no shape tensor in min max node
        value_max_node = utils.create_constant_node(clip_name + '_max_6', [], six)
        value_min_node = utils.create_constant_node(clip_name + '_min_0', [], zero)

        prev_node_names = self.input_nodes_name.copy()
        prev_node_names.append(value_min_node.name)
        prev_node_names.append(value_max_node.name)
        
        clip_node = helper.make_node(
            'Clip',
            inputs=prev_node_names,
            outputs=[clip_name],
            name=clip_name)

        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = helper.make_tensor_value_info(
            clip_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(value_min_node)
        self.node_list.append(value_max_node)
        self.node_list.append(clip_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}

class ClipDefused(ActivationDefused):

    def __init__(self, op, op_type, tflite_interpreter, min_val, max_val):
        ActivationDefused.__init__(self, op, op_type, tflite_interpreter)
        self.min_val = min_val
        self.max_val = max_val

    def generate(self):
        clip_name = self.node_name

        lower = np.array([self.min_val])
        upper = np.array([self.max_val])
        # onnx clip only support no shape tensor in min max node
        value_max_node = utils.create_constant_node(clip_name + '_max_{}'.format(self.max_val), [], upper)
        value_min_node = utils.create_constant_node(clip_name + '_min_{}'.format(self.min_val), [], lower)

        prev_node_names = self.input_nodes_name.copy()
        prev_node_names.append(value_min_node.name)
        prev_node_names.append(value_max_node.name)
        
        clip_node = helper.make_node(
            'Clip',
            inputs=prev_node_names,
            outputs=[clip_name],
            name=clip_name)

        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = helper.make_tensor_value_info(
            clip_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(value_min_node)
        self.node_list.append(value_max_node)
        self.node_list.append(clip_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}


# Normal Activation Layer
class Relu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        relu_node = helper.make_node(
            "Relu",
            name=self.node_name,
            inputs=self.input_nodes_name,
            outputs=[self.node_name]
        )

        # original layer output
        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = onnx.helper.make_tensor_value_info(
            self.node_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(relu_node)

        #Generate Quantization Info and Reverse Quantization for Weights and Bias
        output_quantization_info = node_output_detail["quantization_parameters"]
        output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
        quantization_info = {}
        quantization_info[self.node_name] = output_quantization_info

        return self.node_list, self.value_infos, self.weight_node_list, quantization_info


class Relu6(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        clip_name = self.node_name

        six = np.array([6.0])
        zero = np.array([0.0])
        # onnx clip only support no shape tensor in min max node
        value_max_node = utils.create_constant_node(clip_name + '_max_6', [], six)
        value_min_node = utils.create_constant_node(clip_name + '_min_0', [], zero)

        prev_node_names = self.input_nodes_name.copy()
        prev_node_names.append(value_min_node.name)
        prev_node_names.append(value_max_node.name)

        clip_node = onnx.helper.make_node(
            'Clip',
            inputs=prev_node_names,
            outputs=[clip_name],
            name=clip_name
        )

        # original layer output
        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = onnx.helper.make_tensor_value_info(
            clip_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )

        self.value_infos.append(out_shape_info)
        self.node_list.append(value_min_node)
        self.node_list.append(value_max_node)
        self.node_list.append(clip_node)

        #Generate Quantization Info and Reverse Quantization for Weights and Bias
        output_quantization_info = node_output_detail["quantization_parameters"]
        output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
        quantization_info = {}
        quantization_info[self.node_name] = output_quantization_info

        return self.node_list, self.value_infos, self.weight_node_list, quantization_info


class LOGISTIC(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        logistic_name = self.node_name
        logistic_node = helper.make_node(
            op_type='Sigmoid',
            inputs=self.input_nodes_name,
            outputs=[logistic_name],
            name=logistic_name
        )
        self.node_list.append(logistic_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}


class Softmax(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        softmax_node_name = self.node_name
        softmax_node = onnx.helper.make_node(
            'Softmax',
            inputs=self.input_nodes_name,
            outputs=[softmax_node_name],
            name=softmax_node_name
        )
        self.node_list.append(softmax_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}


class PRelu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        slope_node_info = self.tflite_interpreter._get_tensor_details(self.op.Inputs(1))
        slope_array = self.tflite_interpreter.get_tensor(slope_node_info['index'])
        slope_array = np.transpose(slope_array, (2, 0, 1))

        # make slope onnx node
        slope_onnx_node_name = self.node_name + "_slope"
        slope_onnx_node = onnx.helper.make_tensor(
            slope_onnx_node_name,
            TensorProto.FLOAT,
            slope_array.shape,
            slope_array.flatten().tolist()
        )
        self.weight_node_list.append(slope_onnx_node)

        previous_onnx_node_names = self.input_nodes_name.copy()
        previous_onnx_node_names.extend([slope_onnx_node_name])
        prelu_node = onnx.helper.make_node(
            'PRelu',
            inputs=previous_onnx_node_names,
            outputs=[self.node_name],
            name=self.node_name
        )
        self.node_list.append(prelu_node)

        # original layer output
        node_output_detail = self.tflite_interpreter._get_tensor_details(self.op.Outputs(0))
        out_shape_info = onnx.helper.make_tensor_value_info(
            self.node_name,
            TensorProto.FLOAT,
            utils.tflite2onnx_shape_map(node_output_detail['shape'].tolist())
        )
        self.value_infos.append(out_shape_info)

        #Generate Quantization Info and Reverse Quantization for Weights and Bias
        output_quantization_info = node_output_detail["quantization_parameters"]
        output_quantization_info["dtype"] = str(node_output_detail["dtype"]).split(".")[1].split("'")[0]
        quantization_info = {}
        quantization_info[self.node_name] = output_quantization_info

        return self.node_list, self.value_infos, self.weight_node_list, quantization_info

class Elu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

    def generate(self):
        elu_name = self.node_name
        elu_node = helper.make_node(
            op_type='Elu',
            inputs=self.input_nodes_name,
            outputs=[elu_name],
            name=elu_name
        )
        self.node_list.append(elu_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}

class LeakyRelu(Layer):

    def __init__(self, op, op_type, tflite_interpreter):
        Layer.__init__(self, op, op_type, tflite_interpreter)

        self.tflite_leaky_relu_parser = LeakyReluOptions()
        self.tflite_leaky_relu_parser.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    def generate(self):

        # get  block info
        leaky_relu_alpha = self.tflite_leaky_relu_parser.Alpha()

        # build node
        leaky_relu_name = self.node_name
        leaky_relu_node = helper.make_node(
            'LeakyRelu', 
            self.input_nodes_name, 
            [leaky_relu_name], 
            alpha=leaky_relu_alpha, 
            name=leaky_relu_name 
        )

        # update tables
        self.node_list.append(leaky_relu_node)

        return self.node_list, self.value_infos, self.weight_node_list, {}