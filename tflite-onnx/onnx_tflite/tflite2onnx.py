import onnx 
import onnx.utils
from onnx import helper
from onnx import AttributeProto, TensorProto

import tflite_utils
import logging

import os
from datetime import datetime
import argparse
import tensorflow as tf

from tflite.Model import Model

from tree_structure import Tree

import json 

import math

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

def set_end_node(onnx_end_node, onnx_end_node_shape):

    out_value_info_name = "out_" + onnx_end_node.name
    out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, tflite_utils.tflite2onnx_shape_map(onnx_end_node_shape))

    # change output
    onnx_end_node.output[:] = [out_value_info_name]

    return out_value_info

def build_button_transpose_node_for_channel_first_2_channel_last(onnx_end_node, onnx_end_node_shape):
    transpose_node = None

    out_value_info_name = "out_" + onnx_end_node.name
    out_value_info = helper.make_tensor_value_info( out_value_info_name, TensorProto.FLOAT, onnx_end_node_shape)

    if len(onnx_end_node_shape) == 4:
        # add transpose if it is 4 dimension output

        transpose_node_name = 'transpose_node_bottom_' + onnx_end_node.name
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[onnx_end_node.name],
            outputs=[out_value_info_name],
            perm=[0, 2, 3, 1],
            name=transpose_node_name
        )
    elif len(onnx_end_node_shape) == 3:
        # add transpose if it is 3 dimension output

        transpose_node_name = 'transpose_node_bottom_' + onnx_end_node.name
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

def build_head_transpose_node_for_channel_last_2_channel_first(input_name, transpose_node_name):
    this_node_name = 'transpose_node_head_' + transpose_node_name
    transpose_node = onnx.helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[this_node_name],
        perm=[0, 3, 1, 2],
        name=this_node_name
    )

    return transpose_node

def merge_quantization_info(dumped_info, quantization_info):
    for name in quantization_info:
        if name in dumped_info and ("weight" not in quantization_info[name] and "bias" not in quantization_info[name]):
            continue
        if "quantized_dimension" not in quantization_info[name] or quantization_info[name]["quantized_dimension"] != 0:
            continue
        curr_dict = quantization_info[name]
        if len(curr_dict["scales"]) == 0:
                curr_dict["radix"] = 0
                curr_dict["kneron_scale"] = 0
        else:
            dtype_to_power = {"uint8":8, "int32":32}
            if curr_dict["dtype"] not in dtype_to_power:
                raise TypeError("Unsupported Fix Point Type")
            zero_points = curr_dict["zero_points"]
            scales = curr_dict["scales"]

            mins = [(-1 * zero_points[i]) * scales[i] for i in range(len(zero_points))]
            curr_dict["min"] = mins
            if len(mins) == 1:
                curr_dict["min"] = {"all":mins[0]}
            
            maxs = [((1 << dtype_to_power[curr_dict["dtype"]]) - zero_points[i] - 1) * scales[i] for i in range(len(zero_points))]
            curr_dict["max"] = maxs
            if len(maxs) == 1:
                curr_dict["max"] = {"all":maxs[0]}
            
            def get_radix(scale, max_perchannel, min_perchannel):
                range_tflite = max_perchannel - min_perchannel
                range_kneron = max(abs(max_perchannel), abs(min_perchannel)) * 2
                ratio = range_tflite / range_kneron
                radix = math.floor(math.log(ratio / scale, 2))
                return radix

            # radixs = [int(1 / scales[i]).bit_length() - 1 for i in range(len(zero_points))]
            radixs = [get_radix(scales[i], maxs[i], mins[i]) for i in range(len(zero_points))]
            #radixs = [-int(math.log(scales[i],2)) for i in range(len(zero_points))]
            curr_dict["radix"] = radixs
            if len(radixs) == 1:
                curr_dict["radix"] = {"all":radixs[0]}

            kneron_scales = []
            for i in range(len(zero_points)):
                if radixs[i] >= 0:
                    kneron_scales.append(1 / ((1 << radixs[i]) * scales[i]))
                elif radixs[i] < 0:
                    kneron_scales.append((1 / 2 **(-radixs[i])) * scales[i]) 

            curr_dict["scale"] = kneron_scales
            if len(kneron_scales) == 1:
                curr_dict["scale"] = {"all":kneron_scales[0]}

            if "weight" in quantization_info[name]:
                merge_nested_quantization_info(curr_dict, quantization_info[name]["weight"], "weight")
            if "bias" in quantization_info[name]:
                merge_nested_quantization_info(curr_dict, quantization_info[name]["bias"], "weight") 

        curr_dict["scales"] = curr_dict["scales"].tolist()
        curr_dict["zero_points"] = curr_dict["zero_points"].tolist()

        dumped_info[name] = curr_dict
    
    return 

def merge_nested_quantization_info(dumped_info, quantization_info, name):
    if len(quantization_info["scales"]) == 0:
        quantization_info["radix"] = 0
        quantization_info["kneron_scale"] = 0
    else:
        dtype_to_power = {"uint8":8, "int32":32}
        if quantization_info["dtype"] not in dtype_to_power:
            raise TypeError("Unsupported Fix Point Type")
        zero_points = quantization_info["zero_points"]
        scales = quantization_info["scales"]

        mins = [(-1 * zero_points[i]) * scales[i] for i in range(len(zero_points))]
        quantization_info["min"] = mins
        if len(mins) == 1:
            quantization_info["min"] = {"all":mins[0]}
        
        maxs = [((1 << dtype_to_power[quantization_info["dtype"]]) - zero_points[i]) * scales[i] for i in range(len(zero_points))]
        quantization_info["max"] = maxs
        if len(maxs) == 1:
            quantization_info["max"] = {"all":maxs[0]}
        
        radixs = [int(1 / scales[i]).bit_length() - 1 for i in range(len(zero_points))]
        quantization_info["radix"] = radixs
        if len(radixs) == 1:
            quantization_info["radix"] = {"all":radixs[0]}
        
        kneron_scales = [1 / ((1 << radixs[i]) * scales[i]) for i in range(len(zero_points))]
        quantization_info["scale"] = kneron_scales
        if len(kneron_scales) == 1:
            quantization_info["scale"] = {"all":kneron_scales[0]}

        # quantization_info["min"] = [(-1 * zero_points[i]) * scales[i] for i in range(len(zero_points))]
        # quantization_info["max"] = [((1 << dtype_to_power[quantization_info["dtype"]]) - zero_points[i]) * scales[i] for i in range(len(zero_points))]
        # radix = [int(1 / scales[i]).bit_length() - 1 for i in range(len(zero_points))]
        # quantization_info["radix"] = radix
        # quantization_info["kneron_scale"] = [1 / ((1 << radix[i]) * scales[i]) for i in range(len(zero_points))]
    
    dumped_info[name] = quantization_info
    return 


def check_quantization(tensor_details):
    for node_detail in tensor_details:
        if len(node_detail["quantization_parameters"]["scales"] > 0):
            return True
    return False 

def main(model_path, model_save_path=None, add_transpose_for_channel_last_first_issue = True, bottom_nodes_name = None):

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
    input_tensor_value_info = None

    # generate tree
    tree_graph = Tree(model_path=model_path, bottom_nodes_name=bottom_nodes_name, defused=True)


    # get sequential node name
    sequential_keys = tree_graph.get_sequential_nodes_key()

    # get tree node in the form of {node_name: op_node_obj}
    tree_dict = tree_graph.get_nodes()


    #############################
    # build head transpose node #
    #############################
    for h_node in tree_graph.get_head_nodes():
        # transpose for channel last to channel first
        if add_transpose_for_channel_last_first_issue is True:
            logging.getLogger('tflite2onnx').debug("generating transpose node for channel last first issue: " + h_node.node_name)
            input_tensor_value_info = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, h_node.node_input_shape.tolist())
            h_transpose_node = build_head_transpose_node_for_channel_last_2_channel_first(input_tensor_value_info.name, h_node.node_name)
            
            onnx_node_list.append(h_transpose_node)
            h_node.input_nodes_name = [h_transpose_node.name]
        else:
            input_tensor_value_info = helper.make_tensor_value_info(model_input_name, TensorProto.FLOAT, tflite_utils.tflite2onnx_shape_map(h_node.node_input_shape.tolist()))
            h_node.input_nodes_name = [input_tensor_value_info.name]
                 

    ############################
    # build model node by node #
    ############################
    dumped_quantization_info = {}
    for key in sequential_keys:
        logging.getLogger('tflite2onnx').debug("generating: " + key)
        nodes, val, weight, quantization_info = tree_dict[key].generate()

        if (len(val) != 0) and (tree_dict[key].is_bottom_node is False):
            inner_node_shape_value_info.extend(val)
        if len(weight) != 0:
            onnx_weight_node_list.extend(weight)
        if len(nodes) != 0:
            onnx_node_list.extend(nodes)
        if len(quantization_info) != 0:
            merge_quantization_info(dumped_quantization_info, quantization_info)
            #print(dumped_quantization_info)

    if check_quantization(interpreter.get_tensor_details()): 
        json_save_path = model_save_path[:-5] + "_user_config.json"
        with open (json_save_path, "w") as f:
            print(json_save_path)
            json.dump(dumped_quantization_info, f, indent = 1)
            print("New Qunatized information saved")



    # sometimes, there are sub-node in one tree node, we need to find the last one
    b_nodes = [ node for node in tree_graph.get_bottom_nodes() ]
    
    ###############################
    # build bottom transpose node #
    ###############################
    for b_node in b_nodes:

        out_value_info = None
        if add_transpose_for_channel_last_first_issue is True:
            logging.getLogger('tflite2onnx').debug("generating transpose node for channel last first issue: " + b_node.node_name)
            out_value_info, transpose_node = build_button_transpose_node_for_channel_first_2_channel_last( b_node.node_list[-1], b_node.node_output_shape.tolist() )
            
            if transpose_node != None:
                onnx_node_list.append(transpose_node)
        else:
            out_value_info = set_end_node(b_node.node_list[-1], b_node.node_output_shape.tolist())
        output_tensor_value_info.append(out_value_info)

                


    input_init = [input_tensor_value_info]
    input_init.extend(onnx_weight_node_list)
    onnx_inputs = tflite_utils.make_kneron_valid_onnx_input( input_init )

    graph_cnn = helper.make_graph(
        onnx_node_list,
        'cnn_test',
        onnx_inputs,
        output_tensor_value_info,
        onnx_weight_node_list,
        value_info=inner_node_shape_value_info
    )

    cnn_model = helper.make_model(graph_cnn, producer_name='Kneron')
    cnn_model.opset_import[0].version = 11

    # add generated time to model meta data
    helper.set_model_props(cnn_model, {'Generated Time': datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S") + " (UTC+0)"})

    cnn_model = onnx.utils.polish_model(cnn_model)

    # save
    if model_save_path is not None:
        onnx.save(cnn_model, model_save_path)
    return cnn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file path')
    parser.add_argument('-save_path', metavar='saved model path', help='an output onnx file path')
    parser.add_argument('-bottom_nodes', metavar='bottom node you want', help='nodes name in tflite model which is the bottom node of sub-graph, use "," to add multiple nodes. ex:"con1,softmax2" ')
    parser.add_argument('-release_mode', metavar='is release mode', help='True if no transpose front end needed')
    parser.add_argument('-log_level', metavar='loglevel', help='log level for python logging modul, ex:"-log_level INFO"')
    args = parser.parse_args()

    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    if args.tflite[-7:] != ".tflite":
        raise UserWarning('\n' + args.tflite + " should be .tflite")
    if args.save_path[-5:] != ".onnx":
        raise UserWarning('\n' + args.save_path + " should be .onnx")

    model_path = os.path.abspath(args.tflite)
    model_save_path = os.path.abspath(args.save_path)
    is_release_mode = True if args.release_mode == 'True' else False

    # log level set up
    usr_set_log_level = args.log_level.upper() if args.log_level is not None else "INFO"
    numeric_level = getattr(logging, usr_set_log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)

    # show infomation
    logging.info('-----------   information    ----------------')
    logging.info('is_release_mode: ' + str(is_release_mode))
    logging.info('model_path: ' + model_path)
    logging.info('model_save_path: ' + model_save_path)

    logging.info('-----------    start to generate  -----------')
    logging.info('generating...')


    try:
        bottom_nodes_name = args.bottom_nodes.split(',') if args.bottom_nodes is not None else list()
        main(model_path, model_save_path, not is_release_mode, bottom_nodes_name=bottom_nodes_name)
        logging.getLogger('tflite2onnx').info("Conversion Success")
    except Exception as e:
        logging.getLogger('tflite2onnx').info('Error: Something Wrong')
        logging.getLogger('tflite2onnx').error(e)

    logging.info('------------   end   ------------------------')

