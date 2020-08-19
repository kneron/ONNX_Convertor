import onnx 
import onnx.utils
from onnx import helper
from onnx import AttributeProto, TensorProto

from conv_layers import Convolution,DepthwiseConvolution,ResizeNearestNeighbor,ResizeBilinear,TransposeConvolution
from aact_layers import Relu,Relu6,Softmax,LOGISTIC,PRelu
from core_layers import Dense,Reshape,Pad,Squeeze,L2Normalization
from merg_layers import Add,Mul,Concatenation
from pool_layers import MaxPooling2D,AveragePooling2D,Mean
import utils

import os
from datetime import datetime
import argparse
import json
import tensorflow as tf

import tflite
from tflite.BuiltinOperator import BuiltinOperator
from tflite.Model import Model

from tree_structure import Tree

def read_tflite_model(path):
    data = open(path, "rb").read()
    model = Model.GetRootAsModel(bytearray(data), 0)
    return model
    
def get_op_info(model_path):
    ops = [] 
    op_types = []

    raw_model = read_tflite_model(model_path)
    tflite_graph = raw_model.Subgraphs(0)
    for idx in range(tflite_graph.OperatorsLength()):
        op = tflite_graph.Operators(idx)
        op_type = raw_model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

        ops.append(op)
        op_types.append(op_type)

    return ops, op_types

def set_end_node(onnx_end_node, tflite_out_info):

    out_value_info_name = "out_" + onnx_end_node.name
    out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, utils.tflite2onnx_shape_map(tflite_out_info['shape'].tolist()))

    # change output
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

def main(model_path, model_save_path, add_transpose_for_channel_last_first_issue = True):

    onnx_weight_node_list = []
    output_tensor_value_info = []
    onnx_node_list = []
    inner_node_shape_value_info = []

    # parse node information through tflite interpreter (tflite interpreter can't parse operator information in our target tensorflow version 1.15)
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # get model input info(assume there is only one input)
    input_details = interpreter.get_input_details()
    model_input_name  = input_details[0]['name']
    model_input_shape = input_details[0]['shape'].tolist()
    input_tensor_value_info = None

    # generate tree
    tree_graph = Tree(model_path=model_path, defused=True)

    # get sequential node name
    sequential_keys = tree_graph.get_sequential_nodes_key()

    # get tree node in the form of {node_name: op_node_obj}
    tree_dict = tree_graph.get_nodes()

    if add_transpose_for_channel_last_first_issue is True:
        input_tensor_value_info = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, model_input_shape)
        # transpose for channel last to channel first
        transpose_node = build_head_transpose_node_for_channel_last_2_channel_first(input_tensor_value_info.name)

        # update tables
        onnx_node_list = [transpose_node]

        for root_node in tree_graph.get_head_nodes():
            root_node.input_nodes_name = [transpose_node.name]

    else: 
        onnx_node_list = []
        input_tensor_value_info = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, utils.tflite2onnx_shape_map(model_input_shape))

        for root_node in tree_graph.get_head_nodes():
            root_node.input_nodes_name = [model_input_name]



    ############################
    # build model node by node #
    ############################
    for key in sequential_keys:

        nodes, val, weight = tree_dict[key].generate()

        if len(val) != 0:
            inner_node_shape_value_info.extend(val)
        if len(weight) != 0:
            onnx_weight_node_list.extend(weight)
        if len(nodes) != 0:
            onnx_node_list.extend(nodes)


    
    output_details = interpreter.get_output_details()
    tmp_node_list = onnx_node_list.copy()

    # sometimes, there are sub-node in one tree node, we need to find the last one
    b_nodes_name = [ node.node_list[-1].name for node in tree_graph.get_bottom_nodes() ]
    
    for onnx_node in tmp_node_list:

        # check if it is output node
        if onnx_node.name not in b_nodes_name:
            continue

        for output_node_info in output_details:

            if output_node_info['name'] in onnx_node.name:  # name of defused node is slightly different from original
                output_node_info['name'] = onnx_node.name

                # it's output node
                out_value_info = None
                transpose_node = None
                if add_transpose_for_channel_last_first_issue is True:
                    out_value_info, transpose_node = build_button_transpose_node_for_channel_first_2_channel_last(onnx_node,output_node_info)
                else:
                    out_value_info = set_end_node(onnx_node,output_node_info)
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

    cnn_model = helper.make_model(graph_cnn, producer_name='Kneron')

    # add generated time to model meta data
    helper.set_model_props(cnn_model, {'Generated Time': datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S") + " (UTC+0)"})

    cnn_model = onnx.utils.polish_model(cnn_model)

    # save
    onnx.save(cnn_model, model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file path')
    parser.add_argument('-save_path', metavar='saved model path', help='an output onnx file path')
    parser.add_argument('-release_mode', metavar='is release mode', help='True if no transpose front end needed')
    args = parser.parse_args()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    if args.tflite[-7:] != ".tflite":
        raise UserWarning('\n' + args.tflite + " should be .tflite")
    if args.save_path[-5:] != ".onnx":
        raise UserWarning('\n' + args.save_path + " should be .onnx")

    model_path = os.path.abspath(args.tflite)
    model_save_path = os.path.abspath(args.save_path)
    is_release_mode = True if args.release_mode == 'True' else False




    print('-----------   information    ----------------')
    print('is_release_mode: ' + str(is_release_mode))
    print('model_path: ' + model_path)
    print('model_save_path: ' + model_save_path)

    print('-----------    start to generate  -----------')
    print('generating...')

    try:
        main(model_path, model_save_path, not is_release_mode)
    except Exception as e:
        print('Error: Something Wrong')
        print(e)

    print('------------   end   ------------------------')

