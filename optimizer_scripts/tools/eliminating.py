import collections
import logging
import struct
import onnx
import numpy as np
from . import other
from . import helper
from . import modhelper
from .general_graph import Graph

def eliminate_Identify_and_Dropout(g):
    """
    Eliminate Identify layers

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Identity' and node.op_type != 'Dropout':
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        if value_between is not None:
            g.value_info.remove(value_between)
        # If this node is the last node, add its previous output into the output.
        output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is not None:
            previous_output_value = helper.find_output_by_name(g, node.input[0])
            if previous_output_value is None:
                output_value.name = node.input[0]
            else:
                g.output.remove(output_value)
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)

# Remove last useless nodes
def remove_useless_last_nodes(g):
    """Remove useless nodes from the tail of the graph
    """
    USELESS = ["Reshape", "Identity", "Transpose", "Flatten", "Dropout", "Mystery", "Constant", "Squeeze", "Unsqueeze", 'Softmax']
    graph = Graph(g)
    todo = collections.deque()
    for node in graph.output_nodes:
        if len(node.children) == 0:
            todo.append(node)
    node_to_remove = []
    while todo:
        # BFS find nodes to remove
        cur_node = todo.popleft()
        if cur_node.proto is None:
            continue
        if cur_node.proto.op_type not in USELESS:
            continue
        # Find the output
        cur_node_output = helper.find_output_by_name(g, cur_node.proto.output[0])
        for cur_input in cur_node.parents:
            cur_input.children.remove(cur_node)
            if len(cur_input.children) == 0:
                todo.append(cur_input)
            if cur_node_output is not None:
                cur_input_output = helper.find_value_by_name(g, cur_input.proto.output[0])
                cur_input_output_in_output = helper.find_output_by_name(g, cur_input.proto.output[0])
                if cur_input_output is not None and cur_input_output_in_output is None:
                    g.output.extend([cur_input_output])
        node_to_remove.append(cur_node.proto)
        try:
            g.value_info.remove(helper.find_value_by_name(g, cur_node.proto.output[0]))
        except ValueError:
            pass
        if cur_node_output is not None:
            g.output.remove(cur_node_output)
        cur_node.proto = None
        cur_node.parents.clear()
    for node in node_to_remove:
        g.node.remove(node)

######################################
#  TF only optimization passes       #
######################################

def eliminate_shape_changing_after_input(g):
    """
    Eliminate the Reshape node after input and reshape the input

    :param g: the onnx graph
    """
    node_to_remove = []
    REMOVE_LIST = ["Reshape", "Transpose", "Flatten", "Dropout", "Squeeze", "Unsqueeze"]
    for node in g.node:
        # Here we iterate over nodes to eliminate multiple nodes after input.
        # Find an input and the shape node
        if node.op_type not in REMOVE_LIST:
            continue
        old_input = helper.find_input_by_name(g, node.input[0])
        if old_input is None:
            continue
        # If the input is used by multiple nodes, skip.
        following_nodes = helper.find_following_nodes_by_input_value_name(g, old_input.name)
        if len(following_nodes) > 1:
            continue
        # Find node output value_info
        output_val_info = helper.find_value_by_name(g, node.output[0])

        if node.op_type == 'Reshape':
            shape_node = helper.find_node_by_output_name(g, node.input[1])
            if shape_node.op_type != 'Constant':
                continue

            # Use the output value info as the new input
            if output_val_info is None:
                logging.warn("Cannot eliminate " + node.name + " in the beginning of graph.")
                continue
            new_input = output_val_info

            # Delete node
            node_to_remove.append(node)

            # Delete weight
            shape_outputs = helper.find_nodes_by_input_name(g, shape_node.output[0])
            if len(shape_outputs) <= 1:
                node_to_remove.append(shape_node)
                g.value_info.remove(helper.find_value_by_name(g, shape_node.output[0]))

            # Delete value info
            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == 'Transpose':
            permutation = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            new_shape = [pre_shape[i] for i in permutation]

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == 'Flatten':
            axis = node.attribute[0].int
            pre_shape = helper.get_shape_from_value_info(old_input)
            dim_1, dim_2 = 1, 1
            if axis == 0:
                dim_1 = 1
                dim_2 = np.prod(pre_shape)
            else:
                dim_1 = np.prod(pre_shape[:axis]).astype(int)
                dim_2 = np.prod(pre_shape[axis:]).astype(int)
            new_shape = [dim_1, dim_2]

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == 'Dropout':
            g.input.remove(old_input)
            g.input.extend([output_val_info])
            g.value_info.remove(output_val_info)

            node_to_remove.append(node)
        elif node.op_type == 'Squeeze':
            axis = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            for pos in sorted(axis)[::-1]:
                if pre_shape[pos] != 1:
                    raise RuntimeError('invalid axis for squeeze')
                else:
                    pre_shape.pop(pos)
            new_shape = pre_shape

            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape
            )

            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        elif node.op_type == 'Unsqueeze':
            axis = list(node.attribute[0].ints)
            pre_shape = helper.get_shape_from_value_info(old_input)
            new_shape = pre_shape
            for pos in axis:
                new_shape.insert(pos, 1)
            new_input = onnx.helper.make_tensor_value_info(
                output_val_info.name,
                output_val_info.type.tensor_type.elem_type,
                new_shape
            )
            node_to_remove.append(node)

            g.input.remove(old_input)
            g.input.extend([new_input])
            g.value_info.remove(output_val_info)
        else:
            pass

    for node in node_to_remove:
        g.node.remove(node)
    
    other.topological_sort(g)


def eliminate_Reshape_Cast(g):
    """Eliminate the cast layer for shape of Reshape layer

    :param g: the onnx graph
    """
    #Find all reshape layers
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Reshape':
            continue
        prev_node = helper.find_node_by_output_name(g, node.input[1])
        if prev_node.op_type != 'Cast':
            continue
        # Now we find the cast weight pattern. Cast the weight, delete the cast.
        reshape_node = node
        cast_node = prev_node
        weight_node = helper.find_node_by_output_name(g, cast_node.input[0])
        if weight_node is None:
            raise RuntimeError("Unexpected None before Cast-Reshape.")
        weight_node.attribute[0].t.data_type = 7
        if weight_node.attribute[0].t.raw_data:
            raw_data = weight_node.attribute[0].t.raw_data
            int_data = [i[0] for i in struct.iter_unpack('i', raw_data)]
            raw_data = struct.pack('q' * len(int_data), *int_data)
        elif len(weight_node.attribute[0].t.int64_data) > 0\
            or len(weight_node.attribute[0].t.int32_data) > 0:
            # It's already int. Do nothing
            pass
        else:
            raise NotImplementedError()
        # Change Value info
        origin_weight_out = helper.find_value_by_name(g, weight_node.output[0])
        weight_node.output.pop()
        weight_node.output.extend([reshape_node.input[1]])
        # Delete
        g.value_info.remove(origin_weight_out)
        g.node.remove(cast_node)

def eliminate_Cast_after_input(g):
    """Eliminate the cast layer right after the input

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Cast':
            continue
        old_input = helper.find_input_by_name(g, node.input[0])
        if old_input is None:
            continue
        next_val_info = helper.find_value_by_name(g, node.output[0])
        shape = helper.get_shape_from_value_info(next_val_info)
        new_val_info = onnx.helper.make_tensor_value_info(
            next_val_info.name,
            node.attribute[0].i,
            shape
        )
        # Delete old value_info
        g.input.remove(old_input)
        g.value_info.remove(next_val_info)
        # Append nodes to node_to_remove
        node_to_remove.append(node)
        # Add new input
        g.input.extend([new_val_info])
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_consecutive_Cast(g):
    """If two cast is next to each other, remove the first cast

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Cast':
            continue
        first_node = helper.find_node_by_output_name(g, node.input[0])
        if first_node is None or first_node.op_type != 'Cast':
            continue
        # Here we have two consecutive Cast Node
        # Reset the input of the later node
        node.input[0] = first_node.input[0]
        # Remove the first node and its output value info
        node_to_remove.append(first_node)
        first_output = helper.find_value_by_name(g, first_node.output[0])
        g.value_info.remove(first_output)
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_Squeeze_before_Reshape(g):
    """If Squeeze and Reshape is next to each other, remove the first node

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Reshape':
            continue
        first_node = helper.find_node_by_output_name(g, node.input[0])
        if not first_node:
            continue
        if first_node.op_type != 'Squeeze':
            continue
        # Here we have two consecutive Cast Node
        # Reset the input of the later node
        node.input[0] = first_node.input[0]
        # Remove the first node and its output value info
        node_to_remove.append(first_node)
        first_output = helper.find_value_by_name(g, first_node.output[0])
        g.value_info.remove(first_output)
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_no_children_input(g):
    """Eliminate inputs with no children at all.
    """
    # Create a set of input names
    input_names = set([i.name for i in g.input])
    # If a name is used in any node, remove this name from the set.
    for n in g.node:
        for i in n.input:
            input_names.discard(i)
    # Remove the inputs with the left names.
    for i in input_names:
        info = helper.find_input_by_name(g, i)
        g.input.remove(info)

def eliminate_consecutive_reshape(g):
    """Replace consecutive reshape nodes by a single node.
    """
    RESHAPE_TYPE = set(["Reshape", "Flatten", "Dropout", "Squeeze", "Unsqueeze"])

    for node in g.node:
        # Check if this is a reshape
        if node.op_type not in RESHAPE_TYPE:
            continue
        pre_data_node = helper.find_node_by_output_name(g, node.input[0])
        if not pre_data_node:
            continue
        # Check if the shape is a constant
        if len(node.input) > 1:
            pre_shape_node = helper.find_node_by_output_name(g, node.input[1])
            if not pre_shape_node:
                continue
            if pre_shape_node.op_type != 'Constant':
                continue
        # Check if the previous node is reshape
        if pre_data_node.op_type not in RESHAPE_TYPE:
            continue
        # Check if the weight of the previous node is a constant
        if len(pre_shape_node.input) > 1:
            pre_pre_shape_node = helper.find_node_by_output_name(g, pre_data_node.input[1])
            if pre_pre_shape_node.op_type != 'Constant':
                continue
        # Check if the previous node has only one output connected
        post_pre_nodes = helper.find_nodes_by_input_name(g, node.input[0])
        if len(post_pre_nodes) != 1:
            continue

        #Reconnect the graph
        modhelper.replace_node_input(node, node.input[0], pre_data_node.input[0])

        g.node.remove(pre_data_node)

        val_info_to_del1 = helper.find_value_by_name(g, node.input[0])
        if val_info_to_del1 is not None:
            g.value_info.remove(val_info_to_del1)

def eliminate_single_input_Concat(g):
    """
    Eliminate single input Concat layers

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Concat':
            continue
        # If this node has more than 1 input, continue.
        if len(node.input) > 1:
            continue
        # If this node is the output node, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            the_input_value = helper.find_value_by_name(g, node.input[0])
            g.output.remove(todel_output)
            g.output.extend([the_input_value])
            node_to_remove.append(node)
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except:
            print("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_nop_Maxpool_and_AveragePool(g):
    """
    Eliminate do nothing MaxPool and AveragePool layers.
    Those layers have valid padding, 1x1 kernel and [1,1] strides.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'MaxPool' and node.op_type != 'AveragePool':
            continue
        # If this node is actually working, continue.
        kernel = helper.get_list_attribute_by_name(node, "kernel_shape", "int")
        pads = helper.get_list_attribute_by_name(node, "pads", "int")
        strides = helper.get_list_attribute_by_name(node, "strides", "int")
        if kernel != [1, 1] or pads != [0, 0, 0, 0] or strides != [1, 1]:
            continue
        # If this node is the output node, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            the_input_value = helper.find_value_by_name(g, node.input[0])
            g.output.remove(todel_output)
            g.output.extend([the_input_value])
            node_to_remove.append(node)
            continue
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except:
            print("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_trivial_maxpool(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != 'MaxPool':
            continue
        pads = None
        strides = None
        dilation = None
        kernel_shape = None
        for att in node.attribute:
            if att.name == 'pads':
                pads = list(att.ints)
            elif att.name == 'strides':
                strides = list(att.ints)
            elif att.name == 'kernel_shape':
                kernel_shape = list(att.ints)
            elif att.name == 'dilation':
                dilation = list(att.ints)
            else:
                pass
        if pads and any([pad != 0 for pad in pads]):
            continue
        if strides and any([stride != 1 for stride in strides]):
            continue
        if dilation and any([dila != 1 for dila in dilation]):
            continue
        if any([dim != 1 for dim in kernel_shape]):
            continue

        node_to_del.append(node)

        next_nodes = helper.find_nodes_by_input_name(g, node.output[0])

        if next_nodes[0] == None:
            output_value = helper.find_output_by_name(g, node.output[0])
            if not output_value:
                continue
            else:
                pre_val_info = helper.find_value_by_name(g, node.input[0])
                g.output.extend([pre_val_info])
                g.output.remove(output_value)
        
        for next_node in next_nodes:
            modhelper.replace_node_input(next_node, node.output[0], node.input[0])
        
        next_val_info = helper.find_value_by_name(g, node.output[0])
        g.value_info.remove(next_val_info)

    while node_to_del:
        g.node.remove(node_to_del.pop())
    
    other.topological_sort(g)

def eliminate_empty_value_infos(g):
    to_remove = []
    for value_info in g.value_info:
        if len(value_info.type.tensor_type.shape.dim) == 0:
            to_remove.append(value_info)
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                to_remove.append(value_info)
                break
    for value_info in to_remove:
        g.value_info.remove(value_info)

def eliminate_nop_pads(g):
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Pad':
            continue
        # Check if the Pad is empty or not
        pads_node = helper.find_node_by_output_name(g, node.input[1])
        if pads_node.op_type != 'Constant':
            continue
        pads_np = helper.constant_to_numpy(pads_node)
        all_zero = True
        for value in pads_np:
            if value != 0:
                all_zero = False
        if not all_zero:
            continue
        # Check if it has the constant_value_node
        constant_value_node = None
        if len(node.input) > 2:
            constant_value_node = helper.find_node_by_output_name(g, node.input[2])
        # If this node is the output node, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            g.output.remove(todel_output)
            if helper.find_output_by_name(g, node.input[0]) is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                if the_input_value is not None:
                    g.output.extend([the_input_value])
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        try:
            g.value_info.remove(value_between)
        except:
            helper.logger.info("No value info to delete while eliminating identity layers.")
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_trivial_elementwise_calculation(g):
    """Eliminate Add, Sub, Mul, Sub nodes which do nothing.
    """
    node_to_remove = []
    for node in g.node:
        weight_node = None
        if node.op_type == 'Add' or node.op_type == 'Sub':
            # For add and sub, check if the weights are 0s.
            weight_node = helper.find_node_by_output_name(g, node.input[1])
            if weight_node is None or weight_node.op_type != 'Constant':
                continue
            weight_np = helper.constant_to_numpy(weight_node)
            if np.any(weight_np):
                continue
        elif node.op_type == 'Mul' or node.op_type == 'Div':
            # For Mul and Div, check if the weights are 1s.
            weight_node = helper.find_node_by_output_name(g, node.input[1])
            if weight_node is None or weight_node.op_type != 'Constant':
                continue
            weight_np = helper.constant_to_numpy(weight_node)
            weight_np = weight_np - 1
            if np.any(weight_np):
                continue
        else:
            # For other nodes, just skip
            continue
        # Remove the node
        node_to_remove.append(node)
        output_value_info = helper.find_value_by_name(g, node.output[0])
        if output_value_info is not None:
            g.value_info.remove(output_value_info)
        # Replace next node input if any.
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        todel_output = helper.find_output_by_name(g, node.output[0])
        if todel_output is not None:
            g.output.remove(todel_output)
            previous_output = helper.find_output_by_name(g, node.input[0])
            if previous_output is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                g.output.extend([the_input_value])
        # Delete the constant node if it is not used by other nodes
        constant_following_nodes = helper.find_following_nodes_by_input_value_name(g, weight_node.output[0])
        if len(constant_following_nodes) == 1:
            node_to_remove.append(weight_node)
            output_value_info = helper.find_value_by_name(g, weight_node.output[0])
            if output_value_info is not None:
                g.value_info.remove(output_value_info)
    for node in node_to_remove:
        g.node.remove(node)

def eliminate_nop_cast(g):
    """Eliminate do nothing Cast nodes.
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Cast':
            continue
        # Get input value_info
        input_value = helper.find_value_by_name(g, node.input[0])
        if input_value is None:
            helper.logger.debug(f"Cannot find the input value_info for Cast node {node.name}. Skip elimination check.")
            continue
        # Get output value_info
        output_value = helper.find_value_by_name(g, node.output[0])
        if output_value is None:
            output_value = helper.find_output_by_name(g, node.output[0])
        if output_value is None:
            helper.logger.debug(f"Cannot find the output value_info for Cast node {node.name}. Skip elimination check.")
            continue
        # Compare the type.
        if input_value.type.tensor_type.elem_type != output_value.type.tensor_type.elem_type:
            continue
        # If this node is the output node, set its previous node as output nodes.
        if helper.find_output_by_name(g, node.output[0]) is not None:
            todel_output = helper.find_output_by_name(g, node.output[0])
            g.output.remove(todel_output)
            if helper.find_output_by_name(g, node.input[0]) is None:
                the_input_value = helper.find_value_by_name(g, node.input[0])
                if the_input_value is not None:
                    g.output.extend([the_input_value])
        # Replace the parents in all the following nodes
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # Delete value info
        value_between = helper.find_value_by_name(g, node.output[0])
        if value_between is not None:
            g.value_info.remove(value_between)
        # Node is waiting for elimination
        node_to_remove.append(node)
    for node in node_to_remove:
        g.node.remove(node)


def eliminate_nop_reshape(g):
    for node in g.node:
        # Check if this is a reshape
        if node.op_type != 'Reshape':
            continue
        pre_shape_node = helper.find_node_by_output_name(g, node.input[1])
        # Check if the shape is a constant
        if pre_shape_node.op_type != 'Constant':
            continue
        # Check if the input shape equals the output shape
        input_value_info = helper.find_value_by_name(g, node.input[0])
        output_value_info = helper.find_value_by_name(g, node.output[0])
        if input_value_info is None or output_value_info is None or \
            helper.get_shape_from_value_info(input_value_info) != helper.get_shape_from_value_info(output_value_info):
            continue
        # Connect previous node and the next node
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # delete the node
        g.node.remove(node)
        g.value_info.remove(output_value_info)

def eliminate_nop_flatten(g):
    for node in g.node:
        # Check if this is a reshape
        if node.op_type != 'Flatten':
            continue
        # Check if the input shape equals the output shape
        input_value_info = helper.find_value_by_name(g, node.input[0])
        output_value_info = helper.find_value_by_name(g, node.output[0])
        if input_value_info is None or output_value_info is None or \
            helper.get_shape_from_value_info(input_value_info) != helper.get_shape_from_value_info(output_value_info):
            continue
        # Connect previous node and the next node
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, node.output[0], node.input[0])
        # delete the node
        g.node.remove(node)
        g.value_info.remove(output_value_info)

def get_set_of_connected_node_names(g):
    # Input to output approach
    connected_node_names = set()
    top_names = set([v.name for v in g.input])
    to_visit = collections.deque()
    # Initilize visit queue
    for input_name in top_names:
        input_following_nodes = helper.find_following_nodes_by_input_value_name(g, input_name)
        for input_following_node in input_following_nodes:
            if input_following_node.name in connected_node_names:
                continue
            to_visit.append(input_following_node)
            connected_node_names.add(input_following_node.name)
    # Visit all the nodes in BFS pattern
    while len(to_visit) > 0:
        n = to_visit.popleft()
        for output_name in n.output:
            following_nodes = helper.find_following_nodes_by_input_value_name(g, output_name)
            for following_node in following_nodes:
                if following_node.name in connected_node_names:
                    continue
                to_visit.append(following_node)
                connected_node_names.add(following_node.name)

    # Output to input approach
    to_visit = collections.deque([n for n in g.node if n.name in connected_node_names])


    # Visit all the nodes in BFS pattern
    while len(to_visit) > 0:
        n = to_visit.popleft()
        for input_name in n.input:
            previous_node = helper.find_node_by_output_name(g, input_name)
            if previous_node is None:
                # Constant node or input does not have previous node.
                continue
            if previous_node.name in connected_node_names:
                continue
            to_visit.append(previous_node)
            connected_node_names.add(previous_node.name)

    return connected_node_names


def eliminate_not_connected_nodes(g):
    # Find not connected nodes
    connected_nodes = get_set_of_connected_node_names(g)
    not_connected_node_list = []
    for n in g.node:
        if n.name not in connected_nodes:
            not_connected_node_list.append(n)

    # Remove not connected nodes
    for n in not_connected_node_list:
        logging.debug("Removing not connected: " + n.name)
        g.node.remove(n)

def eliminate_not_connected_outputs(g):
    value_to_remove = []
    for value in g.output:
        node = helper.find_node_by_output_name(g, value.name)
        if node is None:
            value_to_remove.append(value)
    while len(value_to_remove) > 0:
        g.output.remove(value_to_remove.pop())

def clear_value_infos(g):
    while len(g.value_info) != 0:
        g.value_info.pop()

def remove_reshape_of_batch_change(g):
    for n in g.node:
        # Check the operator type
        if n.op_type != 'Reshape':
            continue
        # Get the input and output shape
        input_value = helper.find_value_by_name(g, n.input[0])
        input_shape = helper.get_shape_from_value_info(input_value)
        output_value = helper.find_value_by_name(g, n.output[0])
        output_shape = helper.get_shape_from_value_info(output_value)
        # Check the shape change
        if len(input_shape) == len(output_shape) - 1:
            # 25 x a x b -> 1 x 25 x a x b
            if output_shape[0] != 1:
                continue
            not_matched = False
            for i in range(len(input_shape)):
                if input_shape[i] != output_shape[i + 1]:
                    not_matched = True
                    break
            if not_matched:
                continue
        elif len(input_shape) == len(output_shape) + 1:
            # 1 x 25 x a x b -> 25 x a x b
            if input_shape[0] != 1:
                continue
            not_matched = False
            for i in range(len(output_shape)):
                if input_shape[i + 1] != output_shape[i]:
                    not_matched = True
                    break
            if not_matched:
                continue
        else:
            continue
        # Reconnect the graph
        following_nodes = helper.find_following_nodes_by_input_value_name(g, n.output[0])
        for following_node in following_nodes:
            modhelper.replace_node_input(following_node, n.output[0], n.input[0])
        # Delete the value_info
        g.value_info.remove(output_value)
        # Delete the node
        g.node.remove(n)

def eliminate_Transpose_surround_Concat(g):
    # Find concat
    for concat_node in g.node:
        if concat_node.op_type != 'Concat':
            continue
        # Find all the Transpose before and after
        input_nodes = [helper.find_node_by_output_name(g, input_name) for input_name in concat_node.input]
        failed = False
        for input_node in input_nodes:
            if input_node.op_type != "Transpose":
                failed = True
                break
        if failed:
            continue
        output_nodes = helper.find_following_nodes_by_input_value_name(g, concat_node.output[0])
        for output_node in output_nodes:
            if output_node.op_type != "Transpose":
                failed = True
                break
        if failed:
            continue
        # Change the Concat axis
        axis_attr = helper.get_attribute_by_name(concat_node, 'axis')
        axis_attr.i = 1
        # Reconnect the graph
        while len(concat_node.input) > 0:
            concat_node.input.pop()
        for input_node in input_nodes:
            concat_node.input.append(input_node.input[0])
        for output_node in output_nodes:
            for following_node in helper.find_following_nodes_by_input_value_name(g, output_node.output[0]):
                modhelper.replace_node_input(following_node, output_node.output[0], concat_node.output[0])
        # Delete the tranpose nodes
        for n in input_nodes:
            g.node.remove(n)
        for n in output_nodes:
            g.node.remove(n)

# Reshape Transpose Reshape Transpose pattern elimination
def eliminate_reshape_transpose_pattern(g):
    for transpose_node in g.node:
        # Find Reshape Transpose pattern
        if transpose_node.op_type != 'Transpose':
            continue
        reshape_node = helper.find_node_by_output_name(g, transpose_node.input[0])
        if reshape_node is None:
            continue
        if reshape_node.op_type != 'Reshape':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, transpose_node.input[0])) > 1:
            continue
        # Reconnect
        for child_node in helper.find_following_nodes_by_input_value_name(g, transpose_node.output[0]):
            modhelper.replace_node_input(child_node, transpose_node.output[0], reshape_node.input[0])
        # Delete both nodes
        g.node.remove(reshape_node)
        g.node.remove(transpose_node)
