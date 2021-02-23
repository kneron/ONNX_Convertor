"""This module contains helper functions that do graph modifications.
"""

import onnx
from . import helper


def replace_node_input(node, old_input, new_input):
    for i, input_name in enumerate(node.input):
        if input_name == old_input:
            node.input[i] = new_input

def delete_nodes(g, node_list):
    node_to_delete = []
    #Find target nodes
    for node in g.node:
        if node.name not in node_list:
            continue
        else:
            node_to_delete.append(node)
    if len(node_list) != len(node_to_delete):
        print("Some nodes do not exist in the graph. Skipping them.")
    for node in node_to_delete:
        # Check the node whether if it is valid to delete
        if len(node.input) == 0:
            print("Deleting an Constant node. Please make sure you also delete all its following nodes")
        elif len(node.input) > 1:
            print("Warning: Node {} has more than one input. This script cannot delete merge nodes.".format(node.name))
        # Connect the nodes around the target node.
        # Set the following node input as the previous node output.
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        if len(node.input) == 0:
            for following_node in following_nodes:
                following_node.input.remove(node.output[0])
        elif len(following_nodes) > 0 and len(node.input) == 1 and helper.find_input_by_name(g, node.input[0]) is not None:
            # The node input is an input
            new_input = helper.find_value_by_name(g, node.output[0])
            g.input.append(new_input)
            g.input.remove(helper.find_input_by_name(g, node.input[0]))
            g.value_info.remove(new_input)
        elif len(following_nodes) > 0:
            for following_node in following_nodes:
                replace_node_input(following_node, node.output[0], node.input[0])
        else:
            # If the node is the output, replace the output with the previous input.
            value = helper.find_value_by_name(g, node.input[0])
            output_values = []
            while len(g.output):
                output_values.append(g.output.pop())
            while output_values:
                output_value = output_values.pop()
                if output_value.name == node.output[0]:
                    g.output.extend([value])
                else:
                    g.output.extend([output_value])
        # Remove the node and value info.
        g.node.remove(node)

def delete_input(g, target_list):
    for name in target_list:
        input_value = helper.find_input_by_name(g, name)
        if input_value is None:
            print("Cannot find input {}".format(name))
            continue
        g.input.remove(input_value)

def delete_output(g, target_list):
    for name in target_list:
        output_value = helper.find_output_by_name(g, name)
        if output_value is None:
            print("Cannot find output {}".format(name))
            continue
        g.output.remove(output_value)

def delete_value_with_name_if_exists(g, name):
    value = helper.find_value_by_name(g, name)
    if value is not None:
        g.value_info.remove(value)
