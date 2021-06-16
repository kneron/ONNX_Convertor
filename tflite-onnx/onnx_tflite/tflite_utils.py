import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto
from onnx import onnx_pb as onnx_proto
import numpy as np


def tflite2onnx_shape_map(shape_list):
    # change dimension due to channel first-last issue
    if len(shape_list) == 4:
        return [shape_list[0],shape_list[3],shape_list[1],shape_list[2]]

    elif len(shape_list) == 3:
        return [shape_list[0],shape_list[2],shape_list[1]]

    elif len(shape_list) == 2:
        # not change
        return shape_list

    raise ValueError('unexpected output dimension')


def channel_last_2_channel_first_axis_mapping(axis_list):
    table = {'0':0,
             '3':1,
             '1':2,
             '2':3}
    res = []
    for axis in axis_list:
        res.append(table[str(axis)])
    return res

def get_output_node_info_by_name_if_exist(node_name, interpreter):
    output_details = interpreter.get_output_details()
    for node_info in output_details:
        if node_name == node_info['name']:
            return node_info
    return None

def getPadding(feat_map_size, kernel_size, strides, dilation_factor, mode):
    # only support 'VALID' and 'SAME' mode
    if mode != 'VALID' and mode != 'SAME':
        return None

    # if mode is VALID
    if mode == 'VALID':
        return [0,0,0,0]

    if dilation_factor is None:
        convolution_size = kernel_size
    else:
        convolution_size = [
            kernel_size[0] + ((kernel_size[0] - 1) * (dilation_factor[0] - 1)),
            kernel_size[1] + ((kernel_size[1] - 1) * (dilation_factor[1] - 1))
        ]

    # else, mode is SAME 
    """ Calculate the padding array for same padding in the Tensorflow fashion.
    See https://www.tensorflow.org/api_guides/python/nn#Convolution for more.
    """
    if feat_map_size[1] % strides[0] == 0:
        pad_h = max(convolution_size[0] - strides[0], 0)
    else:
        pad_h = max(convolution_size[0] - feat_map_size[1] % strides[0], 0)

    if feat_map_size[2] % strides[1] == 0:
        pad_w = max(convolution_size[1] - strides[1], 0)
    else:
        pad_w = max(convolution_size[1] - feat_map_size[2] % strides[1], 0)

    return [pad_h//2, pad_w//2, pad_h - pad_h//2, pad_w - pad_w//2]

def make_kneron_valid_onnx_input(input_init):
    onnx_inputs = []
    for data in input_init:

        if isinstance(data, onnx.TensorProto):
            val = helper.make_tensor_value_info(
                data.name, data.data_type.real,
                list(d for d in data.dims))
            onnx_inputs.append(val)

        elif isinstance(data, onnx.AttributeProto):
            value_info = onnx.ValueInfoProto()
            value_info.name = data.name

            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = data.type
            value_info.type.CopyFrom(onnx_type)

            onnx_inputs.append(value_info)
        else:
            onnx_inputs.append(data)
    return onnx_inputs


def get_value_from_dict(dict_obj, key):
    if isinstance(dict_obj, dict):
        if key in dict_obj.keys():
            return dict_obj[key]
        else:
            return None
    else:
        raise TypeError('dict_obj is not type of ' + dict.__name__)

def get_quantization_info_in_array(quantization_info):
    if len(quantization_info["scales"]) == 0:
        quantization_info["radix"] = 0
        quantization_info["kneron_scale"] = 0
    else:
        dtype_to_power = {"uint8":8, "int32":32}
        if quantization_info["dtype"] not in dtype_to_power:
            raise TypeError("Unsupported Fix Point Type")
        zero_points = quantization_info["zero_points"]
        scales = quantization_info["scales"]
        quantization_info["min"] = [(-1 * zero_points[i]) * scales[i] for i in range(len(zero_points))]
        quantization_info["max"] = [((1 << dtype_to_power[quantization_info["dtype"]]) - zero_points[i]) * scales[i] for i in range(len(zero_points))]
        radix = [int(1 / scales[i]).bit_length() - 1 for i in range(len(zero_points))]
        quantization_info["radix"] = radix
        quantization_info["kneron_scale"] = [1 / ((1 << radix[i]) * scales[i]) for i in range(len(zero_points))]
    
    return "1"
def create_constant_node(node_name, shape, data):
    # default data type
    data_type = onnx.helper.TensorProto.FLOAT

    if data.dtype == np.int or data.dtype == np.int64:
        data_type = onnx.helper.TensorProto.INT64

    elif data.dtype == np.float or data.dtype == np.float64:
        data_type = onnx.helper.TensorProto.FLOAT

    else:
        data_type = onnx.helper.TensorProto.FLOAT
    
    value_tensor = onnx.helper.make_tensor(
        node_name, 
        data_type, 
        shape, 
        data.tolist()
    )

    constant_node = onnx.helper.make_node(
        "Constant", 
        [], 
        [node_name], 
        name=node_name, 
        value=value_tensor
    )

    return constant_node
