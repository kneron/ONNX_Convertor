import onnx
from onnx import optimizer
from tools import helper, replacing, other, modhelper
import numpy as np
from extract_loop import multiply_Loop_graph_and_replace

m = onnx.load("/home/kneron/Downloads/model_se_loop/model_se.1.onnx")
g = m.graph


# Replace shape and folllowing nodes.
# Create new node.
zeros_128_np = np.zeros((1, 128), dtype=np.float32)
zeros_128_node = helper.numpy_to_constant('sequential_1/lstm_1/zeros', zeros_128_np)
# Remove old nodes.
name_of_node_to_remove = set([
    'sequential_1/lstm_1/Shape',
    'sequential_1/lstm_1/Shape__34',
    'sequential_1/lstm_1/strided_slice',
    'sequential_1/lstm_1/zeros/packed_Concat__44',
    'sequential_1/lstm_1/zeros__45',
    'sequential_1/lstm_1/zeros'
])
node_to_remove = []
for node in g.node:
    if node.name in name_of_node_to_remove:
        node_to_remove.append(node)
for node in node_to_remove:
    g.node.remove(node)
# Add new nodes
g.node.insert(0, zeros_128_node)


# Remove TensorListFromTensor node and replace input
g.input[0].type.tensor_type.shape.dim[0].dim_value = 40
g.input[0].type.tensor_type.shape.dim[1].dim_value = 1
g.input[0].type.tensor_type.shape.dim.pop()
# Remove old nodes.
name_of_node_to_remove = set([
    'transpose',
    'TensorArrayUnstack/TensorListFromTensor'
])
node_to_remove = []
for node in g.node:
    if node.name in name_of_node_to_remove:
        node_to_remove.append(node)
for node in node_to_remove:
    g.node.remove(node)
# Replace input
loop_node = helper.find_node_by_node_name(g, 'while_loop')
loop_node.input[7] = 'serving_default_lstm_1_input:0'


# Expand Loop
multiply_Loop_graph_and_replace(g, loop_node, 40)
other.topological_sort(m.graph)


# Replace TensorListReserve and TensorListSetItem
# Create concat node
concat_input = []
new_nodes = []
for node in g.node:
    if node.op_type == 'TensorListSetItem':
        unsqueeze_node = onnx.helper.make_node(
            op_type = 'Unsqueeze',
            inputs = [node.input[2]],
            outputs = [node.output[0]],
            name = node.name,
            axes = [0]
        )
        concat_input.append(node.output[0])
        new_nodes.append(unsqueeze_node)
concat_node = onnx.helper.make_node(
    op_type = 'Concat',
    inputs = concat_input,
    outputs = ['while_loop:2'],
    name = 'while_loop:2',
    axis = 0
)
new_nodes.append(concat_node)
following_node = helper.find_node_by_node_name(g, 'strided_slice_23')
modhelper.replace_node_input(following_node, 'loop_39_while/TensorArrayV2Write/TensorListSetItem', 'while_loop:2')
# Remove old nodes.
node_to_remove = []
for node in g.node:
    if node.op_type in ['TensorListSetItem', 'TensorListReserve']:
        node_to_remove.append(node)
for node in node_to_remove:
    g.node.remove(node)
# Add new nodes
g.node.extend(new_nodes)


# Fix squeeze for opset 12
# Create new node.
original_node = helper.find_node_by_node_name(g, 'strided_slice_23__52')
squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [original_node.input[0]],
            outputs = [original_node.output[0]],
            name = original_node.name,
            axes = [0]
)
# Remove old nodes.
g.node.remove(original_node)
# Add new nodes
g.node.append(squeeze_node)


# Fix Gather in opset 12
# Create concat node
new_nodes = []
node_to_remove = []
for node in g.node:
    if node.op_type == 'Gather':
        gather_node = onnx.helper.make_node(
            op_type = 'Gather',
            inputs = node.input,
            outputs = node.output,
            name = node.name,
            axis = 0
        )
        new_nodes.append(gather_node)
        node_to_remove.append(node)
# Remove old nodes.
for node in node_to_remove:
    g.node.remove(node)
# Add new nodes
g.node.extend(new_nodes)
other.topological_sort(g)

# Save
onnx.save(m, "/home/kneron/Downloads/model_se_loop/model_se.2.onnx")