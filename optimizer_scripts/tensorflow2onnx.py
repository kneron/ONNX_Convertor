import tensorflow as tf
import tf2onnx
import argparse
import logging
import sys
import onnx.utils
from tensorflow.python.platform import gfile
from tools import combo, eliminating, replacing

TF2ONNX_VERSION = int(tf2onnx.version.version.replace('.', ''))

if 160 <= TF2ONNX_VERSION:
    from tf2onnx import tf_loader
else:
    from tf2onnx import loader as tf_loader

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Convert tensorflow pb file to onnx file and optimized onnx file. Or just optimize tensorflow onnx file.')
parser.add_argument('in_file', help='input file')
parser.add_argument('out_file', help='output optimized model file')
parser.add_argument('-t', '--test_mode', default=False, help='test mode will not eliminate shape changes after input')

args = parser.parse_args()
input_path = args.in_file
output_path = args.out_file

if args.in_file[-3:] == '.pb':
    model_name = args.in_file.split('/')[-1][:-3]
    input_path = args.in_file[:-3] + '.onnx'

    with tf.Session() as sess:
        with gfile.FastGFile(args.in_file, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        if 160 <= int(tf2onnx.version.version.replace('.', '')):
            onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes, functions = tf2onnx.tf_utils.tflist_to_onnx(
                sess.graph,
                {})
        else:
            onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tf2onnx.tfonnx.tflist_to_onnx(
                sess.graph.get_operations(),
                {})

        for n in onnx_nodes:
            if len(n.output) == 0:
                onnx_nodes.remove(n)

        # find inputs and outputs of graph
        nodes_names = [n.name for n in onnx_nodes]
        nodes_inputs = set()
        nodes_outputs = set()

        for n in onnx_nodes:
            if n.op_type == 'Placeholder':
                continue
            for input in n.input:
                nodes_inputs.add(input)
            for output in n.output:
               nodes_outputs.add(output)

        graph_input_names = set()
        for input_name in nodes_inputs:
            if input_name not in nodes_outputs:
                graph_input_names.add(input_name)

        graph_output_names = set()
        for n in onnx_nodes:
            if n.input and n.input[0] not in nodes_outputs:
                continue
            if len(n.output) == 0:
                n.output.append(n.name + ':0')
                graph_output_names.add(n.output[0])
            else:
                output_name = n.output[0]
                if (output_name not in nodes_inputs) and (0 < len(n.input)):
                    graph_output_names.add(output_name)

    logging.info('Model Inputs: %s', str(list(graph_input_names)))
    logging.info('Model Outputs: %s', str(list(graph_output_names)))

    graph_def, inputs, outputs = tf_loader.from_graphdef(model_path=args.in_file,
                                                         input_names=list(graph_input_names),
                                                         output_names=list(graph_output_names))

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')

    if 160 <= TF2ONNX_VERSION:
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph=tf_graph,
                                                         input_names=inputs,
                                                         output_names=outputs,
                                                         opset=11)
    else:
        with tf.Session(graph=tf_graph):
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph=tf_graph,
                                                         input_names=inputs,
                                                         output_names=outputs,
                                                         opset=11)

    # Optimize with tf2onnx.optimizer
    onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)
    model_proto = onnx_graph.make_model(model_name)

    # Make tf2onnx output compatible with the spec. of onnx.utils.polish_model
    replacing.replace_initializer_with_Constant(model_proto.graph)
    model_proto = onnx.utils.polish_model(model_proto)
    
else:
    raise Exception('expect .pb file as input, but got "' + str(args.in_file) + '"')

# rename
m = model_proto 

m = combo.preprocess(m)
m = combo.common_optimization(m)
m = combo.tensorflow_optimization(m)
m = combo.postprocess(m)

if not args.test_mode:
    g = m.graph
    eliminating.eliminate_shape_changing_after_input(g)

m = onnx.utils.polish_model(m)

onnx.save(m, output_path)

logging.info('Save Optimized ONNX: %s', output_path)

