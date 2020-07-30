from collections import defaultdict
import numpy as np
import onnx.helper
import onnx.utils

from . import modhelper
from . import helper
from . import other

def torch_pattern_match(m):
    # Create a map from optype to the nodes.
    optype2node = defaultdict(list)
    for node in m.graph.node:
        optype2node[node.op_type].append(node)
    for matmul_node in optype2node['MatMul']:
        pattern_matmul_mul_add(m.graph, matmul_node)
    m = onnx.utils.polish_model(m)
    return m

def pattern_matmul_mul_add(g, matmul_node):
    # Check node match - Mul node
    next_nodes = helper.find_nodes_by_input_name(g, matmul_node.output[0])
    if len(next_nodes) != 1:
        return
    if next_nodes[0].op_type != 'Mul':
        return
    mul_node = next_nodes[0]
    # Check node match - Add node
    next_nodes = helper.find_nodes_by_input_name(g, mul_node.output[0])
    if len(next_nodes) != 1:
        return
    if next_nodes[0].op_type != 'Add':
        return
    add_node = next_nodes[0]
    # Check Mul weight
    mul_weight_node = helper.find_node_by_output_name(g, mul_node.input[1])
    if mul_weight_node.op_type != 'Constant':
        return
    weight_size, mul_weight = helper.constant_to_list(mul_weight_node)
    for i in mul_weight:
        if i != 1:
            return
    channel = weight_size[0]
    # Check Add weight
    add_weight_node = helper.find_node_by_output_name(g, add_node.input[1])
    if add_weight_node.op_type != 'Constant':
        return
    # Check MatMul weight to see if it need weight broadcast
    matmul_weight_node = helper.find_node_by_output_name(g, matmul_node.input[1])
    matmul_weight = helper.constant_to_numpy(matmul_weight_node)
    if matmul_weight.shape[1] == 1:
        # Weight broadcast
        new_matmul_weight = np.tile(matmul_weight, channel)
        new_matmul_weight_node = helper.numpy_to_constant(matmul_weight_node.name, new_matmul_weight)
        g.node.remove(matmul_weight_node)
        g.node.extend([new_matmul_weight_node])
    value = helper.find_value_by_name(g, matmul_weight_node.output[0])
    if value is not None:
        g.value_info.remove(value)
    # Remove Mul node
    g.node.remove(mul_weight_node)
    value = helper.find_value_by_name(g, mul_weight_node.output[0])
    if value is not None:
        g.value_info.remove(value)
    g.node.remove(mul_node)
    value = helper.find_value_by_name(g, mul_node.output[0])
    if value is not None:
        g.value_info.remove(value)
    # Fuse Matmul and Add
    gemm_node = onnx.helper.make_node(
        'Gemm',
        [matmul_node.input[0], matmul_node.input[1], add_node.input[1]],
        [add_node.output[0]],
        name = matmul_node.name,
        alpha = 1.0,
        beta = 1.0,
        transA = 0,
        transB = 0
    )
    g.node.extend([gemm_node])
    # Clean up
    g.node.remove(matmul_node)
    g.node.remove(add_node)
    value = helper.find_value_by_name(g, matmul_node.output[0])
    if value is not None:
        g.value_info.remove(value)
    other.topological_sort(g)
