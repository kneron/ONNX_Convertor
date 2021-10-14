import numpy as np

from tools import other, helper
from onnx import optimizer
import sys
import onnx
import onnx.helper

def transpose_matmaul_weight(g):
    for matmul_node in g.node:
        if matmul_node.op_type != 'MatMul':
            continue
        output_value = helper.find_value_by_name(g, matmul_node.output[0])
        if output_value is not None:
            continue
        # Create new weight
        weight_node = helper.find_node_by_output_name(g, matmul_node.input[1])
        old_weight = helper.constant_to_numpy(weight_node)
        new_weight = np.transpose(old_weight)
        new_weight_node = helper.numpy_to_constant(matmul_node.input[1], new_weight)
        # Delete old weight
        g.node.remove(weight_node)
        weight_output_value = helper.find_value_by_name(g, matmul_node.input[1])
        if weight_output_value is not None:
            g.value_info.remove(weight_output_value)
        # Add new weight
        g.node.append(new_weight_node)

m = onnx.load(sys.argv[1])
transpose_matmaul_weight(m.graph)
other.topological_sort(m.graph)
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, sys.argv[2])