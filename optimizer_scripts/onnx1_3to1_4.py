# ref http://192.168.200.1:8088/jiyuan/converter_scripts.git

import sys
import onnx
import numpy as np
from onnx import numpy_helper
from tools import other, helper

"""
Change onnx model from version 1.3 to version 1.4. 
Modify the BN node by removing the spatial attribute
Modify the Upsample node by removing the 'scales' attribute, and adding a constant node instead.
Model's ir_version and opset_import are updated.
"""

def remove_BN_spatial(g):
    for node in g.node:
        if node.op_type != 'BatchNormalization':
            continue
        for att in node.attribute:
            if att.name == 'spatial':
                node.attribute.remove(att)


def upsample_attribute_to_const(g):
    for node in g.node:
        if node.op_type != 'Upsample':
            continue
        scales_exist = False
        for att in node.attribute:
            if att.name == 'scales':
                scales_exist = True
                break
        if not scales_exist:
            continue

        shape = [len(att.floats)]
        node.attribute.remove(att)
        new_node = helper.list_to_constant(node.name+'_input', shape, att.floats)

        g.node.extend([new_node])
        value_info = onnx.helper.make_tensor_value_info(node.name+'_input', onnx.TensorProto.FLOAT, shape)
        node.input.extend([node.name+'_input'])
        g.value_info.extend([value_info])

def relu6_to_clip(g):
    for node in g.node:
        if node.op_type != 'Relu':
            continue
        max_val = helper.get_var_attribute_by_name(node, 'max', 'float')
        if max_val is None:
            continue
        new_node = onnx.helper.make_node(
            "Clip",
            node.input,
            node.output,
            name=node.name,
            max=max_val,
            min=0.0
        )
        g.node.remove(node)
        g.node.extend([new_node])

def PRelu_weight_reshape(g):
    # For PRelu with single dimension weight. Expand it to 1, x, 1, 1
    for node in g.node:
        if node.op_type != "PRelu":
            continue
        slope = helper.find_node_by_output_name(g, node.input[1])
        if slope is not None:
            # Constant node
            if len(slope.attribute[0].t.dims) != 1:
                continue
            slope.attribute[0].t.dims.append(slope.attribute[0].t.dims[0])
            slope.attribute[0].t.dims[0] = 1
            slope.attribute[0].t.dims.append(1)
            slope.attribute[0].t.dims.append(1)
        else:
            # Initializer
            for i in g.initializer:
                if i.name == node.input[1]:
                    slope = i
                    break
            if len(slope.dims) != 1:
                continue
            slope.dims.append(slope.dims[0])
            slope.dims[0] = 1
            slope.dims.append(1)
            slope.dims.append(1)
            input_value = helper.find_input_by_name(g, node.input[1])
            new_input = onnx.helper.make_tensor_value_info(
                node.input[1],
                input_value.type.tensor_type.elem_type,
                (1, slope.dims[1], 1, 1))
            g.input.remove(input_value)
            g.input.append(new_input)
        value_info = helper.find_value_by_name(g, node.input[1])
        if value_info is not None:
            g.value_info.remove(value_info)

def do_convert(m):
    graph = m.graph
    
    # Modify the nodes.
    remove_BN_spatial(graph)
    upsample_attribute_to_const(graph)
    relu6_to_clip(graph)
    PRelu_weight_reshape(graph)
    other.topological_sort(graph)

    # Change model properties.
    m.ir_version = 4
    m.opset_import[0].version = 9
    return m

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:{} file_in file_out".format(sys.argv[0]))
        exit(1)

    model = onnx.load(sys.argv[1])
    graph = model.graph

    # Modify the nodes.
    remove_BN_spatial(graph)
    upsample_attribute_to_const(graph)
    relu6_to_clip(graph)
    PRelu_weight_reshape(graph)
    other.topological_sort(graph)

    # Change model properties.
    model.ir_version = 4
    model.opset_import[0].version = 9

    onnx.save(model, sys.argv[2])
