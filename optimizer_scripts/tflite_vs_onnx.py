import argparse
import numpy as np
import tensorflow as tf
import onnx
import onnxruntime

from tools import helper

def compare_tflite_and_onnx(tflite_file, onnx_file, total_times=10):
    # Setup onnx session and get meta data
    onnx_session = onnxruntime.InferenceSession(onnx_file, None)
    onnx_outputs = onnx_session.get_outputs()
    assert len(onnx_outputs) == 1, "The onnx model has more than one output"
    onnx_model = onnx.load(onnx_file)
    onnx_graph = onnx_model.graph
    onnx_inputs = onnx_graph.input
    assert len(onnx_inputs) == 1, "The onnx model has more than one input"
    _, onnx_input_shape = helper.find_size_shape_from_value(onnx_inputs[0])
    # Setup TFLite sessio and get meta data
    tflite_session = tf.lite.Interpreter(model_path=tflite_file)
    tflite_session.allocate_tensors()
    tflite_inputs = tflite_session.get_input_details()
    tflite_outputs = tflite_session.get_output_details()
    tflite_input_shape = tflite_inputs[0]['shape']
    # Compare input shape
    assert(len(onnx_input_shape) == len(tflite_input_shape)), "TFLite and ONNX shape unmatch."
    assert(onnx_input_shape == [tflite_input_shape[0], tflite_input_shape[3], tflite_input_shape[1], tflite_input_shape[2]]), "TFLite and ONNX shape unmatch."
    # Generate random number and run
    tflite_results = []
    onnx_results = []
    for _ in range(total_times):
        # Generate input
        tflite_input_data = np.array(np.random.random_sample(tflite_input_shape), dtype=np.float32)
        onnx_input_data = np.transpose(tflite_input_data, [0, 3, 1, 2])
        # Run tflite
        tflite_session.set_tensor(tflite_inputs[0]['index'], tflite_input_data)
        tflite_session.invoke()
        tflite_results.append(tflite_session.get_tensor(tflite_outputs[0]['index']))
        # Run onnx
        onnx_input_dict = {onnx_inputs[0].name: onnx_input_data}
        onnx_results.append(onnx_session.run([], onnx_input_dict)[0])

    return tflite_results, onnx_results


if __name__ == '__main__':
    # Argument parser.
    parser = argparse.ArgumentParser(description="Compare a TFLite model and an ONNX model to check if they have the same output.")
    parser.add_argument('tflite_file', help='input tflite file')
    parser.add_argument('onnx_file', help='input ONNX file')

    args = parser.parse_args()

    results_a, results_b = compare_tflite_and_onnx(args.tflite_file, args.onnx_file, total_times=10)
    ra_flat = helper.flatten_with_depth(results_a, 0)
    rb_flat = helper.flatten_with_depth(results_b, 0)
    shape_a = [item[1] for item in ra_flat]
    shape_b = [item[1] for item in rb_flat]
    assert shape_a == shape_b, 'two results data shape doesn\'t match'
    ra_raw = [item[0] for item in ra_flat]
    rb_raw = [item[0] for item in rb_flat]

    try:
        np.testing.assert_almost_equal(ra_raw, rb_raw, 8)
        print('Two models have the same behaviour.')
    except Exception as mismatch:
        print(mismatch)
        exit(1)