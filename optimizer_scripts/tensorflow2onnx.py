import tensorflow as tf 
import tf2onnx
import argparse
import onnx.utils
from tensorflow.python.platform import gfile
from tools import combo
from tools import eliminating

file_path = '../models/tensorflow/mnist.pb'

parser = argparse.ArgumentParser(description='Convert tensorflow pb file to onnx file and optimized onnx file. Or just optimize tensorflow onnx file.')
parser.add_argument('in_file', help='input file')
parser.add_argument('out_file', help='output optimized model file')
parser.add_argument('-t', '--test_mode', default=False, help='test mode will not eliminate shape changes after input')

args = parser.parse_args()
input_path = args.in_file
output_path = args.out_file

if args.in_file[-3:] == '.pb':
  model_name = args.in_file.split('/')[-1][:-3]
  input_path = args.in_file[:-3]+'.onnx'

  with tf.Session() as sess:
    with gfile.FastGFile(args.in_file, 'rb') as f:
      graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    onnx_nodes, op_cnt, attr_cnt, output_shapes, dtypes = tf2onnx.tfonnx.tflist_to_onnx(sess.graph.get_operations(), {})
    
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
        n.output.append(n.name+':0')
        graph_output_names.add(n.output[0])
      else:
        output_name = n.output[0]
        if output_name not in nodes_inputs:
          graph_output_names.add(output_name)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
      sess.graph,
      input_names=list(graph_input_names),
      output_names=list(graph_output_names)
    )
    model_proto = onnx_graph.make_model(model_name)
    output_onnx_path = '/'.join(args.out_file.split('/')[:-1])+'/'+model_name+'.onnx'
    
    with open(output_onnx_path, 'wb') as f:
      f.write(model_proto.SerializeToString())
    
    m = onnx.load(output_onnx_path)
elif args.in_file[-5:] == '.onnx':
    m = onnx.load(input_path)

m = combo.preprocess(m)
m = combo.common_optimization(m)
m = combo.tensorflow_optimization(m)
m = combo.postprocess(m)

if not args.test_mode:
  g = m.graph
  eliminating.eliminate_shape_changing_after_input(g)
  m = onnx.utils.polish_model(m)

onnx.save(m, output_path)

