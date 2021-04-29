# ref http://192.168.200.1:8088/jiyuan/converter_scripts.git

import sys
import onnx
import onnx.utils
import numpy as np
from onnx import numpy_helper
from tools import other, helper, replacing

"""
Change onnx model from version 1.4 to version 1.6.
"""

def replace_all_attribute_to_const_node_in_pad_node(g):
    node_to_remove = []
    node_to_extend = []
    for node in g.node:
        if node.op_type != 'Pad':
            continue

        pad_loc_node = None # must have
        pad_mode = 'constant'
        pad_value_node = helper.list_to_constant(node.name+'_pad_value',  [], [0.0])  # need scalar
        for att in node.attribute:
            if att.name == 'mode':
                pad_mode = helper.get_var_attribute_by_name(node, 'mode', 'string')
            if att.name == 'pads':
                pad_loc_node = helper.list_to_constant(node.name+'_pad_loc',  [len(att.ints)], att.ints)  
            if att.name == 'value':
                pad_value_node = helper.list_to_constant(node.name+'_pad_value',  [], [att.f])     

        new_node = onnx.helper.make_node(
            "Pad",
            [node.input[0], pad_loc_node.name, pad_value_node.name],
            [node.output[0]],
            name=node.output[0],
            mode=pad_mode,
        )
        node_to_remove.append(node)
        node_to_extend.append(new_node)
        node_to_extend.append(pad_loc_node)
        node_to_extend.append(pad_value_node)
    
    for node in  node_to_remove:
        g.node.remove(node)
    for node in  node_to_extend:
        g.node.extend([node])


def upsampling_to_resize(g):
    for node in g.node:
        if node.op_type != 'Upsample':
            continue
        upsampling_mode = helper.get_var_attribute_by_name(node, 'mode', 'string')

        scale_value_node = helper.find_node_by_output_name(g, node.input[1])
        if scale_value_node.op_type != "Constant":
            raise TypeError('seems there is a dynamic "scales" param in Upsampling node: ' + node.name + ' , you might need to do constant folding first')

        roi_node = helper.list_to_constant(node.name+'_roi_value',  [0], [])  

        new_node = onnx.helper.make_node(
            "Resize",
            [node.input[0], roi_node.name, scale_value_node.name],
            [node.output[0]],
            name=node.output[0],
            mode=upsampling_mode,
            coordinate_transformation_mode = 'asymmetric'
        )

        g.node.remove(node)
        g.node.extend([new_node])
        g.node.extend([roi_node])


def replace_all_attribute_to_const_node_in_slice_node(g):
    for node in g.node:
        if node.op_type != 'Slice':
            continue

        axes_const_node = None
        ends_const_node = None
        starts_const_node = None
        steps_const_node = None
        for att in node.attribute:
            if att.name == 'axes':
                axes_const_node = helper.list_to_constant(node.name+'_axes_value',  [len(att.ints)], att.ints)     
    
            if att.name == 'ends':
                ends_const_node = helper.list_to_constant(node.name+'_ends_value', [len(att.ints)], att.ints)  

            if att.name == 'starts':
                starts_const_node = helper.list_to_constant(node.name+'_starts_value', [len(att.ints)], att.ints)  

            if att.name == 'steps':
                steps_const_node = helper.list_to_constant(node.name+'_steps_value',[ len(att.ints)], att.ints)

        ## pop out from back 
        attr_len = len(node.attribute)
        for i in range(attr_len):
            node.attribute.remove(node.attribute[ attr_len -1 - i ]) 

        ## according the spec, we need to add node in specific order
        if starts_const_node != None: 
            g.node.extend([starts_const_node])
            node.input.extend([starts_const_node.name])
        if ends_const_node != None: 
            g.node.extend([ends_const_node])
            node.input.extend([ends_const_node.name])        
        if axes_const_node != None: 
            g.node.extend([axes_const_node])
            node.input.extend([axes_const_node.name])
        if steps_const_node != None: 
            g.node.extend([steps_const_node])
            node.input.extend([steps_const_node.name])
        

def replace_min_max_attribute_to_const_node_in_clip_node(g):
    for node in g.node:
        if node.op_type != 'Clip':
            continue

        max_const_node = None
        min_const_node = None
        for att in node.attribute:
            if att.name == 'max':
                max_const_node = helper.list_to_constant(node.name+'_max_value',  [], [att.f])     
    
            if att.name == 'min':
                min_const_node = helper.list_to_constant(node.name+'_min_value', [], [att.f])  

        ## pop out from back 
        node.attribute.remove(node.attribute[1]) 
        node.attribute.remove(node.attribute[0])     
        
        ## according the spec, we need to add node in specific order
        g.node.extend([min_const_node])
        g.node.extend([max_const_node])
        node.input.extend([min_const_node.name])
        node.input.extend([max_const_node.name])

def onnx1_4to1_6(model: onnx.ModelProto) -> onnx.ModelProto:
    """Update ir_version from 4 to 6 and update opset from 9 to 11.

    Args:
        model (onnx.ModelProto): input onnx model.

    Returns:
        onnx.ModelProto: updated onnx model.
    """
    graph = model.graph

    if model.opset_import[0].version == 11:
        print("(Stop) the input model is already opset 11, no need to upgrade")
        exit(1)

    # deal with empty node name issue
    other.add_name_to_node(graph)
    # simplify the node param type from initializer to constant
    replacing.replace_initializer_with_Constant(graph)

    # Modify the nodes.
    replace_min_max_attribute_to_const_node_in_clip_node(graph)
    replace_all_attribute_to_const_node_in_slice_node(graph)
    replace_all_attribute_to_const_node_in_pad_node(graph)
    upsampling_to_resize(graph)
    other.topological_sort(graph)

    # Change model properties.
    model.ir_version = 6
    model.opset_import[0].version = 11

    model = onnx.utils.polish_model(model)
    return model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:{} file_in file_out".format(sys.argv[0]))
        exit(1)

    model = onnx.load(sys.argv[1])
    model = onnx1_4to1_6(model)

    onnx.save(model, sys.argv[2])
