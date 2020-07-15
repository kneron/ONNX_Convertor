import onnx 
import onnx.utils
from onnx import helper
from onnx import AttributeProto, TensorProto

from conv_layers import Convolution,DepthwiseConvolution
from aact_layers import Relu,Relu6,Softmax,LOGISTIC
from core_layers import Dense,Reshape,Pad,Squeeze
from merg_layers import Add,Mul,Concatenation
from pool_layers import MaxPooling2D,AveragePooling2D,Mean
import utils

import os
import argparse
import json
import tensorflow as tf


def set_end_node(onnx_end_node, tflite_out_info):

    out_value_info = None
    out_value_info_name = "out_" + onnx_end_node.name
    if len(tflite_out_info['shape'].tolist()) == 4:
        out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, utils.tflite2onnx_shape_map(tflite_out_info['shape'].tolist()))
    elif len(tflite_out_info['shape'].tolist()) == 2:
        out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, tflite_out_info['shape'].tolist())
    else:
        raise ValueError('unexpected output dimension')
    onnx_end_node.output[:] = [out_value_info_name]

    return out_value_info

def build_button_transpose_node_for_channel_first_2_channel_last(onnx_end_node, tflite_out_info):
    transpose_node = None

    out_value_info_name = "out_" + onnx_end_node.name
    out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, tflite_out_info['shape'].tolist())

    if len(tflite_out_info['shape'].tolist()) == 4:
        # add transpose if it is 4 dimension output

        transpose_node_name = 'transpose_node_output_' + onnx_end_node.name
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[onnx_end_node.name],
            outputs=[out_value_info_name],
            perm=[0, 2, 3, 1],
            name=transpose_node_name
        )
    elif len(tflite_out_info['shape'].tolist()) == 3:
        # add transpose if it is 3 dimension output

        transpose_node_name = 'transpose_node_output_' + onnx_end_node.name
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[onnx_end_node.name],
            outputs=[out_value_info_name],
            perm=[0, 2, 1],
            name=transpose_node_name
        )
    else:
        # no need transpose, set it as output
        onnx_end_node.output[:] = [out_value_info_name]

    return out_value_info, transpose_node

def get_op_info_from_json(model_json_path):
    op_types = []

    json_data = json.load(open(model_json_path))
    for element in json_data['operator_codes']:
        if 'builtin_code' in element.keys():
            op_types.append(element['builtin_code'])
    ops = json_data['subgraphs'][0]['operators']

    return ops, op_types

def build_head_transpose_node_for_channel_last_2_channel_first(input_name):
    transpose_node_name = 'transpose_node_input_' + input_name
    transpose_node = onnx.helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[transpose_node_name],
        perm=[0, 3, 1, 2],
        name=transpose_node_name
    )
    return transpose_node

def main(model_path, model_json_path, model_save_path, add_transpose_for_channel_last_first_issue = True):
    ops, op_types = get_op_info_from_json(model_json_path)

    # some nodes are merged as one node, we need a table to store this information
    op_name__sub_op_name__table = {}

    onnx_weight_node_list = []
    output_tensor_value_info = []
    onnx_node_list = []
    inner_node_shape_value_info = []

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # get input info
    input_details = interpreter.get_input_details()
    input_tensor_value_info = None

    if add_transpose_for_channel_last_first_issue is True:
        input_tensor_value_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_details[0]['shape'].tolist())
        # transpose for channel last to channel first
        transpose_node = build_head_transpose_node_for_channel_last_2_channel_first(input_tensor_value_info.name)

        # update tables
        onnx_node_list = [transpose_node]
        op_name__sub_op_name__table[input_details[0]['name']] = [input_details[0]['name'],transpose_node.name]   
    else: 
        onnx_node_list = []
        input_tensor_value_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, utils.tflite2onnx_shape_map(input_details[0]['shape'].tolist()))
        op_name__sub_op_name__table[input_details[0]['name']] = [input_details[0]['name'],input_tensor_value_info.name]  


    ############################
    # build model node by node #
    ############################
    for op in ops:
        node_output_detail = interpreter._get_tensor_details(op['outputs'][0])
        node_input_detail = interpreter._get_tensor_details(op['inputs'][0])

        node_name = node_output_detail['name'] 
        prev_node_name = node_input_detail['name']

        if prev_node_name in op_name__sub_op_name__table:
            prev_node_name = op_name__sub_op_name__table[prev_node_name][-1] # last sub node

        op_type = op_types[op['opcode_index']]
        if op_type == 'CONV_2D':
            nodes, val, weight = Convolution( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'DEPTHWISE_CONV_2D':
            nodes, val, weight = DepthwiseConvolution( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'SOFTMAX':
            nodes, val, weight = Softmax( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'RELU':
            nodes, val, weight = Relu( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'RELU6':
            nodes, val, weight = Relu6( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'LOGISTIC':
            nodes, val, weight = LOGISTIC( [prev_node_name], op_type, op, interpreter).generate()       
        elif op_type == 'FULLY_CONNECTED':
            nodes, val, weight = Dense( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'RESHAPE':
            nodes, val, weight = Reshape( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'PAD':
            nodes, val, weight = Pad( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'ADD':
            nodes, val, weight = Add( '', op_type, op, interpreter).generate(op_name__sub_op_name__table)
        elif op_type == 'MUL':
            nodes, val, weight = Mul( '', op_type, op, interpreter).generate(op_name__sub_op_name__table)
        elif op_type == 'CONCATENATION':
            nodes, val, weight = Concatenation( '', op_type, op, interpreter).generate(op_name__sub_op_name__table)
        elif op_type == 'MEAN':
            nodes, val, weight = Mean( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'MAX_POOL_2D':
            nodes, val, weight = MaxPooling2D( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'AVERAGE_POOL_2D':
            nodes, val, weight = AveragePooling2D( [prev_node_name], op_type, op, interpreter).generate()
        elif op_type == 'SQUEEZE':
            nodes, val, weight = Squeeze( [prev_node_name], op_type, op, interpreter).generate()
        else:
            raise ValueError(op_type)

        sub_op_node_list = []
        for node in nodes:
            if node.op_type != 'Constant':
                sub_op_node_list.append(node)

        # update tables        
        op_name__sub_op_name__table[node_name] = [sub_op_node.name for sub_op_node in sub_op_node_list]

        if len(val) != 0:
            inner_node_shape_value_info.extend(val)
        if len(weight) != 0:
            onnx_weight_node_list.extend(weight)
        if len(nodes) != 0:
            onnx_node_list.extend(nodes)

        # check if it is output node use original node name
        output_node_info = utils.get_output_node_info_by_name_if_exist(node_name, interpreter)

        if output_node_info is not None:
            # it's output node
            out_value_info = None
            transpose_node = None
            if add_transpose_for_channel_last_first_issue is True:
                out_value_info, transpose_node = build_button_transpose_node_for_channel_first_2_channel_last(sub_op_node_list[-1],output_node_info)
            else:
                out_value_info = set_end_node(sub_op_node_list[-1],output_node_info)
            output_tensor_value_info.append(out_value_info)
            if transpose_node != None: 
                onnx_node_list.append(transpose_node)




    input_init = [input_tensor_value_info]
    input_init.extend(onnx_weight_node_list)
    onnx_inputs = utils.make_kneron_valid_onnx_input( input_init )

    graph_cnn = helper.make_graph(
        onnx_node_list,
        'cnn_test',
        onnx_inputs,
        output_tensor_value_info,
        onnx_weight_node_list,
        value_info=inner_node_shape_value_info
    )

    cnn_model = helper.make_model(graph_cnn, producer_name='onnx-tflite-examples')
    if add_transpose_for_channel_last_first_issue is True:
        cnn_model = onnx.utils.polish_model(cnn_model)
    onnx.save(cnn_model, model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file')
    parser.add_argument('-save_path', metavar='saved model path', help='an output folder path')
    parser.add_argument('-release_mode', metavar='is release mode', help='True if no traspose front end needed')
    args = parser.parse_args()

    model_path = args.tflite
    model_json_path = "./" + os.path.basename(model_path[:-7]) + ".json"
    model_save_path = os.path.abspath(args.save_path) + '/' +os.path.basename(model_path[:-7]) + ".onnx"
    is_release_mode = True if args.release_mode == 'True' else False

    os.system("./flatc/flatc -t --strict-json --defaults-json -o ./ ./flatc/schema.fbs -- " + model_path)

    main(model_path, model_json_path, model_save_path, not is_release_mode)
    os.system("rm " + model_json_path)
