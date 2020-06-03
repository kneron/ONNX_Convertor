import onnxruntime
import onnx
import argparse
import numpy as np
from tools import helper


onnx2np_dtype = {0: 'float', 1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16', 6: 'int32', 7: 'int64', 8: 'str', 9: 'bool', 10: 'float16', 11: 'double', 12: 'uint32', 13: 'uint64', 14: 'complex64', 15: 'complex128', 16: 'float'}


def onnx_model_results(path_a, path_b, total_times=10):
    """ using onnxruntime to inference two onnx models' ouputs
    
    :onnx model paths: two model paths
    :total_times: inference times, default to be 10
    :returns: inference results of two models
    """
    # load model a and model b to runtime
    session_a = onnxruntime.InferenceSession(path_a, None)
    session_b = onnxruntime.InferenceSession(path_b, None)
    outputs_a = session_a.get_outputs()
    outputs_b = session_b.get_outputs()

    # check outputs
    assert len(outputs_a) == len(outputs_b), 'Two models have different output numbers.'
    for i in range(len(outputs_a)):
        out_shape_a, out_shape_b = outputs_a[i].shape, outputs_b[i].shape
        out_shape_a = list(map(lambda x: x if type(x) == type(1) else 1, out_shape_a))
        out_shape_b = list(map(lambda x: x if type(x) == type(1) else 1, out_shape_b))
        assert out_shape_a == out_shape_b, 'Output {} has unmatched shapes'.format(i)


    # load onnx graph_a and graph_b, to find the initializer and inputs
    # then compare to remove the items in the inputs which will be initialized
    model_a, model_b = onnx.load(path_a), onnx.load(path_b)
    graph_a, graph_b = model_a.graph, model_b.graph
    inputs_a, inputs_b = graph_a.input, graph_b.input
    init_a, init_b = graph_a.initializer, graph_b.initializer

    # remove initializer from raw inputs
    input_names_a, input_names_b = set([ele.name for ele in inputs_a]), set([ele.name for ele in inputs_b])
    init_names_a, init_names_b = set([ele.name for ele in init_a]), set([ele.name for ele in init_b])
    real_inputs_names_a, real_inputs_names_b = input_names_a - init_names_a, input_names_b - init_names_b

    # prepare and figure out matching of real inputs a and real inputs b
    # try to keep original orders of each inputs
    real_inputs_a, real_inputs_b = [], []
    for item in inputs_a:
        if item.name in real_inputs_names_a:
            real_inputs_a.append(item)
    for item in inputs_b:
        if item.name in real_inputs_names_b:
            real_inputs_b.append(item)

    # suppose there's only one real single input tensor for each model
    # find the real single inputs for model_a and model_b
    real_single_input_a = None
    real_single_input_b = None
    size_a, size_b = 0, 0
    shape_a, shape_b = [], []
    for item_a in real_inputs_a:
        size, shape = helper.find_size_shape_from_value(item_a)
        if size:
            assert real_single_input_a is None, 'Multiple inputs of first model, single input expected.'
            real_single_input_a = item_a
            size_a, shape_a = size, shape
    for item_b in real_inputs_b:
        size, shape = helper.find_size_shape_from_value(item_b)
        if size:
            assert real_single_input_b is None, 'Multiple inputs of second model, single input expected.'
            real_single_input_b = item_b
            size_b, shape_b = size, shape
    assert size_a == size_b, 'Sizes of two models do not match.'


    # construct inputs tensors
    input_data_type_a = real_single_input_a.type.tensor_type.elem_type
    input_data_type_b = real_single_input_b.type.tensor_type.elem_type
    input_data_type_a = onnx2np_dtype[input_data_type_a]
    input_data_type_b = onnx2np_dtype[input_data_type_b]

    # run inference
    times = 0
    results_a = [[] for i in range(len(outputs_a))]
    results_b = [[] for i in range(len(outputs_b))]
    while times < total_times:
        # initialize inputs by random data, default to be uniform 
        data = np.random.random(size_a)
        input_a = np.reshape(data, shape_a).astype(input_data_type_a)
        input_b = np.reshape(data, shape_b).astype(input_data_type_b)

        input_dict_a = {}
        input_dict_b = {}
        for item_a in real_inputs_a:
            item_type_a = onnx2np_dtype[item_a.type.tensor_type.elem_type]
            input_dict_a[item_a.name] = np.array([]).astype(item_type_a) \
                if item_a.name != real_single_input_a.name else input_a
        for item_b in real_inputs_b:
            item_type_b = onnx2np_dtype[item_b.type.tensor_type.elem_type]
            input_dict_b[item_b.name] = np.array([]).astype(item_type_b) \
                if item_b.name != real_single_input_b.name else input_b

        ra = session_a.run([], input_dict_a)
        rb = session_b.run([], input_dict_b)
        for i in range(len(outputs_a)):
            results_a[i].append(ra[i])
            results_b[i].append(rb[i])
        times += 1

    return results_a, results_b

if __name__ == '__main__':
    # Argument parser.
    parser = argparse.ArgumentParser(description="Compare two ONNX models to check if they have the same output.")
    parser.add_argument('in_file_a', help='input ONNX file a')
    parser.add_argument('in_file_b', help='input ONNX file b')

    args = parser.parse_args()

    results_a, results_b = onnx_model_results(args.in_file_a, args.in_file_b, total_times=10)
    ra_flat = helper.flatten_with_depth(results_a, 0)
    rb_flat = helper.flatten_with_depth(results_b, 0)
    shape_a = [item[1] for item in ra_flat]
    shape_b = [item[1] for item in rb_flat]
    assert shape_a == shape_b, 'two results data shape doesn\'t match'
    ra_raw = [item[0] for item in ra_flat]
    rb_raw = [item[0] for item in rb_flat]

    try:
        np.testing.assert_almost_equal(ra_raw, rb_raw, 4)
        print('Two models have the same behaviour.')
    except Exception as mismatch:
        print(mismatch)
        exit(1)
