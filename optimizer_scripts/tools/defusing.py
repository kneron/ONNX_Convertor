import onnx
import numpy as np
from . import helper
from .other import topological_sort
from .modhelper import delete_value_with_name_if_exists, replace_node_input

def defuse_Einsum(g):
    # Check for Einsum
    """
    Replace Reshape node into Flatten node if applicable.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Einsum':
            continue
        if len(node.input) > 3:
            helper.logger.error(f"Currently do not support Einsum node with more than 3 inputs: {node.name}")
            exit(1)
        equation = helper.get_var_attribute_by_name(node, 'equation', 'string')
        equation_list = equation.split('->')
        inputs = equation_list[0].split(',')
        if len(equation_list) != 2:
            helper.logger.error(f"Currently do not support classical Einsum node: {node.name}")
            exit(1)
        if '...' in equation_list[0]:
            helper.logger.error(f"Currently do not support Einsum node with broadcasting: {node.name}")
            exit(1)
        if len(inputs) == 2:
            new_nodes = defuse_Einsum_2_inputs(inputs[0], inputs[1], equation_list[1],
                                                node.input[0], node.input[1], node.output[0])
            g.node.extend(new_nodes)
            node_to_remove.append(node)
        else:
            new_nodes = defuse_Einsum_2_inputs(inputs[0], inputs[1], equation_list[1],
                                                node.input[0], node.input[1], node.output[0] + '_defuse_phase_0')
            g.node.extend(new_nodes)
            new_nodes = defuse_Einsum_2_inputs(equation_list[1], inputs[2], equation_list[1],
                                                node.output[0] + '_defuse_phase_0', node.input[2], node.output[0])
            g.node.extend(new_nodes)
            node_to_remove.append(node)

    for node in node_to_remove:
        g.node.remove(node)


def defuse_Einsum_2_inputs(input_a_str, input_b_str, output_str, input_a_name, input_b_name, output_name):
    # For 2 inputs Einsum, do matching.
    new_nodes = []
    output_str_list = list(output_str)
    input_a_str_list = list(input_a_str)
    input_b_str_list = list(input_b_str)
    # Find the index to sum on
    sum_letters = []
    for i in input_a_str:
        if i in input_b_str and i not in output_str:
            sum_letters.append(i)
            index_of_i_in_a = input_a_str.index(i)
            index_of_i_in_b = input_b_str.index(i)
            if index_of_i_in_a == 0 and index_of_i_in_b ==0:
                output_str_list.insert(0, '*')
            elif index_of_i_in_a == 0:
                output_str_list.insert(index_of_i_in_b, '*')
            elif index_of_i_in_b == 0:
                output_str_list.insert(index_of_i_in_a, '*')
            else:
                prev_char_in_a = input_a_str[index_of_i_in_a - 1]
                prev_char_in_b = input_b_str[index_of_i_in_b - 1]
                index_prev_a = output_str.index(prev_char_in_a)
                index_prev_b = output_str.index(prev_char_in_b)
                output_str_list.insert(max(index_prev_a, index_prev_b) + 1, '*')
    # Expand inputs
    for i in range(len(output_str_list)):
        i_o = output_str_list[i]
        # A
        if i >= len(input_a_str_list):
            input_a_str_list.append('#')
        elif i_o == input_a_str_list[i]:
            # Matched
            pass
        elif i_o == '*' and input_a_str_list[i] in sum_letters:
            # Matched
            pass
        else:
            input_a_str_list.insert(i, '#')
        # B
        if i >= len(input_b_str_list):
            input_b_str_list.append('#')
        elif i_o == input_b_str_list[i]:
            # Matched
            pass
        elif i_o == '*' and input_b_str_list[i] in sum_letters:
            # Matched
            pass
        else:
            input_b_str_list.insert(i, '#')
    # Create Unsqueeze Node if needed
    if '#' in input_a_str_list:
        unsqueeze_a_name = input_a_name + '_unsqueezed'
        unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [input_a_name],
                outputs = [unsqueeze_a_name],
                name = unsqueeze_a_name,
                axes = [i for i, x in enumerate(input_a_str_list) if x == "#"]
            )
        new_nodes.append(unsqueeze_node)
    else:
        unsqueeze_a_name = input_a_name
    if '#' in input_b_str_list:
        unsqueeze_b_name = input_b_name + '_unsqueezed'
        unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [input_b_name],
                outputs = [unsqueeze_b_name],
                name = unsqueeze_b_name,
                axes = [i for i, x in enumerate(input_b_str_list) if x == "#"]
        )
        new_nodes.append(unsqueeze_node)
    else:
        unsqueeze_b_name = input_b_name
    # Create Mul Node
    if '*' in output_str_list:
        mul_name = output_name + '_internal_mul'
    else:
        mul_name = output_name
    mul_node = onnx.helper.make_node(
        op_type = 'Mul',
        inputs = [unsqueeze_a_name, unsqueeze_b_name],
        outputs = [mul_name],
        name = mul_name
    )
    new_nodes.append(mul_node)
    # Create ReduceSum node if needed
    if '*' in output_str_list:
        reducesum_node = onnx.helper.make_node(
                op_type = 'ReduceSum',
                inputs = [mul_name],
                outputs = [output_name],
                name = output_name,
                axes = [i for i, x in enumerate(output_str_list) if x == "*"],
                keepdims = 0
        )
        new_nodes.append(reducesum_node)
    return new_nodes


