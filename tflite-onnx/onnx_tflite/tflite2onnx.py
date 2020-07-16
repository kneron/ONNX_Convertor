import onnx 
import onnx.utils
from onnx import helper
from onnx import AttributeProto, TensorProto

from conv_layers import Convolution,DepthwiseConvolution
from aact_layers import Relu,Relu6,Softmax,LOGISTIC
from core_layers import Dense,Reshape,Pad
from merg_layers import Add
from pool_layers import MaxPooling2D,AveragePooling2D,Mean
import utils

import os
import argparse
import json
import tensorflow as tf


def check_end_node(node, interpreter):

    output_tensor_value_info = None
    # if is output node, change this node's output 
    output_node_info = utils.get_output_node_info_by_name_if_exist(node.name,interpreter)
    if output_node_info != None:
        out_value_info_name = node.name
        out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, output_node_info['shape'].tolist())
        output_tensor_value_info = out_value_info
        node.output[:] = [node.name]

    return output_tensor_value_info


def get_op_info_from_json(model_json_path):
    op_types = []

    json_data = json.load(open(model_json_path))
    for element in json_data['operator_codes']:
        if 'builtin_code' in element.keys():
            op_types.append(element['builtin_code'])
    ops = json_data['subgraphs'][0]['operators']

    return ops, op_types




def main(model_path, model_json_path, model_save_path, add_transpose_for_channel_last_first_issue = False):
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
    input_tensor_value_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_details[0]['shape'].tolist())

    if add_transpose_for_channel_last_first_issue:
        # transpose for channel last to channel first
        transpose_node_name = 'transpose_node_input'
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[input_tensor_value_info.name],
            outputs=[transpose_node_name],
            perm=[0, 3, 1, 2],
            name=transpose_node_name
        )

        # update tables
        onnx_node_list = [transpose_node]
        op_name__sub_op_name__table[input_details[0]['name']] = [input_details[0]['name'],transpose_node_name]   
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
        elif op_type == 'MEAN':
            nodes, val, weight = Mean( [prev_node_name], op_type, op, interpreter).generate()
        else:
            raise ValueError(op_type)

        sub_op_name_list = []
        for node in nodes:
            if node.op_type != 'Constant':
                sub_op_name_list.append(node.name)

        # update tables        
        op_name__sub_op_name__table[node_name] = sub_op_name_list
        out_value_info = check_end_node(nodes[-1],interpreter)
        if out_value_info != None:
            output_tensor_value_info.append(out_value_info)
        if len(val) != 0:
            inner_node_shape_value_info.extend(val)
        if len(weight) != 0:
            onnx_weight_node_list.extend(weight)
        if len(nodes) != 0:
            onnx_node_list.extend(nodes)


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
    cnn_model = onnx.utils.polish_model(cnn_model)
    onnx.save(cnn_model, model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file')
    parser.add_argument('-save_path', metavar='saved model path', help='an output onnx file')
    args = parser.parse_args()

    model_path = args.tflite
    model_json_path = "./" + os.path.basename(model_path[:-7]) + ".json"
    model_save_path = os.path.abspath(args.save_path) + '/' +os.path.basename(model_path[:-7]) + ".onnx"

    os.system("./flatc/flatc -t --strict-json --defaults-json -o ./ ./flatc/schema.fbs -- " + model_path)

    main(model_path, model_json_path, model_save_path)
    os.system("rm " + model_json_path)
