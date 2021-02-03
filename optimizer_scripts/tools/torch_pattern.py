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
    for resize_node in optype2node['Resize']:
        # torch nn.UpsamplingBilinear2d will give us 4 input: "X, roi, scales, sizes"
        if len(resize_node.input) != 4:
            continue
        make_UpsamplingBilinear2d_value_info(m.graph, resize_node.name)
        m = onnx.shape_inference.infer_shapes(m)
        polish_RESIZE_input_param_node(m.graph, resize_node.name)
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

def make_UpsamplingBilinear2d_value_info(g, resize_node_name):
    resize_node = helper.find_node_by_output_name(g, resize_node_name)

    shape_data_node = helper.find_node_by_output_name(g, resize_node.input[3])
    shape_data = helper.constant_to_numpy(shape_data_node).astype(int)
    l_shape_data = list(shape_data)
    if l_shape_data[0] == 0:
        l_shape_data[0] = 1 + l_shape_data[0]
    shape_data = np.array(l_shape_data)

    new_output_value_info = onnx.helper.make_tensor_value_info(
        resize_node.output[0],
        onnx.helper.TensorProto.FLOAT,
        shape_data.tolist()
    )

    g.value_info.extend([new_output_value_info])

def polish_RESIZE_input_param_node(g, resize_node_name):
    resize_node = helper.find_node_by_output_name(g, resize_node_name)

    shape_data_node = helper.find_node_by_output_name(g, resize_node.input[3])
    shape_data = helper.constant_to_numpy(shape_data_node).astype(int)
    
    # handle 0 batch size which is invalid 
    if shape_data[0] == 0:
        shape_data[0] = 1

    pre_node_output_value_info = helper.find_value_by_name(g, resize_node.input[0])
    ori_shape = np.array([pre_node_output_value_info.type.tensor_type.shape.dim[0].dim_value,
                    pre_node_output_value_info.type.tensor_type.shape.dim[1].dim_value,
                    pre_node_output_value_info.type.tensor_type.shape.dim[2].dim_value,
                    pre_node_output_value_info.type.tensor_type.shape.dim[3].dim_value])
    
    resize_node.input.remove(resize_node.input[3])
    

    resize_scales = np.array(shape_data/ori_shape).astype(float)
    resize_scale_node = helper.list_to_constant('resize_scales_node_' + resize_node.name, resize_scales.shape, resize_scales, data_type=onnx.helper.TensorProto.FLOAT)

    resize_node.input[2] = resize_scale_node.name
    g.node.extend([resize_scale_node])
    
    other.topological_sort(g)
