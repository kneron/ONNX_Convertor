import onnx
import onnx.helper
import numpy as np
from . import helper, modhelper
from .other import topological_sort

def defuse_Einsum(g):
    """
    Defuse Einsum node into Mul and ReduceSum.

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        # Check for Einsum
        if node.op_type != 'Einsum':
            continue
        if len(node.input) > 3:
            helper.logger.error(f"Currently do not support Einsum node with more than 3 inputs: {node.name}")
            exit(1)
        equation = helper.get_var_attribute_by_name(node, 'equation', 'string')
        equation_list = equation.split('->')
        # print(equation_list)
        inputs = equation_list[0].split(',')
        if len(equation_list) != 2:
            helper.logger.error(f"Currently do not support classical Einsum node: {node.name}")
            exit(1)
        if '...' in equation_list[0]:
            helper.logger.error(f"Currently do not support Einsum node with broadcasting: {node.name}")
            exit(1)
        # print(node)
        if len(inputs) == 2:
            new_nodes = defuse_Einsum_2_inputs(inputs[0], inputs[1], equation_list[1],
                                                node.input[0], node.input[1], node.output[0])
            g.node.extend(new_nodes)
            node_to_remove.append(node)
        else:
            # print("abnormal inputs")
            # print(inputs)
            # print(equation_list)
            new_nodes = defuse_Einsum_2_inputs(inputs[0], inputs[1], equation_list[1],
                                                node.input[0], node.input[1], node.output[0] + '_defuse_phase_0')
            g.node.extend(new_nodes)
            new_nodes = defuse_Einsum_2_inputs(equation_list[1], inputs[2], equation_list[1],
                                                node.output[0] + '_defuse_phase_0', node.input[2], node.output[0])
            g.node.extend(new_nodes)
            node_to_remove.append(node)
        # print(new_nodes)

    for node in node_to_remove:
        g.node.remove(node)


def defuse_Einsum_2_inputs(input_a_str, input_b_str, output_str, input_a_name, input_b_name, output_name):
    # For 2 inputs Einsum, do matching.
    new_nodes = []
    
    # output_str_list = list(output_str)
    # input_a_str_list = list(input_a_str)
    # input_b_str_list = list(input_b_str)
    output_str_list = output_str.split()
    input_a_str_list = input_a_str.split()
    input_b_str_list = input_b_str.split()

    print("original shapes!")
    print("A:", input_a_str_list)
    print("B:", input_b_str_list)
    print("C:", output_str_list)

    # Find the index to sum on
    sum_letters = []
    # for i in input_a_str_list:
    #     if i in input_a_str_list and i not in output_str_list:
    #         print("*:'"+i+"'")
    #         sum_letters.append(i)
    #         index_of_i_in_a = input_a_str_list.index(i)
    #         index_of_i_in_b = input_b_str_list.index(i)
    #         print(index_of_i_in_a, index_of_i_in_b)

    #         if index_of_i_in_a == 0 and index_of_i_in_b ==0:
    #             output_str_list.insert(0, '*')
    #         elif index_of_i_in_a == 0:
    #             output_str_list.insert(index_of_i_in_b, '*')
    #         elif index_of_i_in_b == 0:
    #             output_str_list.insert(index_of_i_in_a, '*')
    #         else:
    #             prev_char_in_a = input_a_str_list[index_of_i_in_a - 1]
    #             prev_char_in_b = input_b_str_list[index_of_i_in_b - 1]
    #             print(prev_char_in_a, prev_char_in_b)
    #             index_prev_a = output_str.index(prev_char_in_a)
    #             index_prev_b = output_str.index(prev_char_in_b)
    #             print(prev_char_in_a, prev_char_in_b)
    #             print(index_prev_a, index_prev_b)
    #             output_str_list.insert(max(index_prev_a, index_prev_b) + 1, '*')
    full_str_list = list(input_b_str_list)
    # Get full ranks of the equation
    for i in input_a_str_list:
        if i in input_a_str_list and i not in input_b_str_list:
            print("*:'"+i+"'")
            index_of_i_in_a = input_a_str_list.index(i)
            prev_char_in_a = input_a_str_list[index_of_i_in_a - 1]
            print(index_of_i_in_a, prev_char_in_a)
            index_prev_b = input_b_str_list.index(prev_char_in_a)
            full_str_list.insert(index_prev_b + 1, i)
            input_b_str_list.insert(index_prev_b + 1, '#')
            break
    print("full:", full_str_list)
    # Gxpand inputs and outputs
    for i in full_str_list:
        if i not in input_a_str_list:
            index_of_i_in_full = full_str_list.index(i)
            input_a_str_list.insert(index_of_i_in_full, '#')
        if i not in output_str_list:
            index_of_i_in_full = full_str_list.index(i)
            output_str_list.insert(index_of_i_in_full, '*')
    
    # # Expand inputs
    # for i in range(len(output_str_list)):
    #     i_o = output_str_list[i]
    #     # A
    #     if i >= len(input_a_str_list):
    #         input_a_str_list.append('#')
    #     elif i_o == input_a_str_list[i]:
    #         # Matched
    #         pass
    #     elif i_o == '*' and input_a_str_list[i] in sum_letters:
    #         # Matched
    #         pass
    #     else:
    #         input_a_str_list.insert(i, '#')
    #     # B
    #     if i >= len(input_b_str_list):
    #         input_b_str_list.append('#')
    #     elif i_o == input_b_str_list[i]:
    #         # Matched
    #         pass
    #     elif i_o == '*' and input_b_str_list[i] in sum_letters:
    #         # Matched
    #         pass
    #     else:
    #         input_b_str_list.insert(i, '#')
    
    # new_input_a_str_list = []
    # new_input_b_str_list = []
    # for i in range(len(output_str_list)):
    #     i_o = output_str_list[i]
    #     # A
    #     if i_o in input_a_str_list:
    #         new_input_a_str_list.append(i_o)
    #     else:
    #         new_input_a_str_list.append('#')
    #     # B
    #     if i_o in new_input_b_str_list:
    #         new_input_b_str_list.append(i_o)
    #     else:
    #         new_input_b_str_list.append('#')

    print("modified shapes!")
    print("A:", input_a_str_list)
    print("B:", input_b_str_list)
    print("C:", output_str_list)

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


def defuse_ReduceSum(g):
    """
    Defuse ReduceSum node into ReduceSum and Squeeze.

    :param g: the onnx graph
    """
    for node in g.node:
        # Check for ReduceSum
        if node.op_type != 'ReduceSum':
            continue
        keepdims = helper.get_var_attribute_by_name(node, 'keepdims', "int")
        if keepdims is None or keepdims == 1:
            continue
        # Create new nodes
        internal_output_name = node.name + '_unsqueezed_internal'
        attribute = helper.get_attribute_by_name(node, 'keepdims')
        attribute.i = 1
        origin_output = node.output.pop()
        node.output.append(internal_output_name)
        squeeze_node = onnx.helper.make_node(
                op_type = 'Squeeze',
                inputs = [internal_output_name],
                outputs = [origin_output],
                name = node.name + '_squeezed_internal',
                axes = helper.get_list_attribute_by_name(node, 'axes', 'int')
        )
        g.node.append(squeeze_node)


def defuse_div_with_reciprocal_and_mul(g):
    """
    Defuse Div with Reciprocal and Mul.

    Args:
        g (GraphProto): the graph to process
    """
    node_to_remove = []
    for node in g.node:
        # Find Div node
        if node.op_type != 'Div':
            continue
        # Check if the second input is not constant
        second_input = helper.find_initializer_by_name(g, node.input[1])
        if second_input is not None:
            continue
        second_input = helper.find_node_by_output_name(g, node.input[1])
        if second_input is not None and second_input.op_type == 'Constant':
            continue
        # Construct new nodes.
        reciprocal_node = onnx.helper.make_node(
            "Reciprocal",
            [node.input[1]],
            [node.input[1] + '_reciprocal'],
            name=node.input[1] + '_reciprocal'
        )
        mul_node = onnx.helper.make_node(
            'Mul',
            [node.input[0], node.input[1] + '_reciprocal'],
            node.output,
            name=node.name
        )
        # Construct new Mul node.
        g.node.extend([reciprocal_node, mul_node])
        node_to_remove.append(node)
    # Remove old nodes
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)


def defuse_Conv3D(g):
    """
    Defuse Conv3D with special kernel into Conv2D.

    Args:
        g (GraphProto): the graph to process
    """
    node_to_remove = []
    for node in g.node:
        # Find Conv3D node
        if node.op_type != 'Conv':
            continue
        input_shape = helper.get_shape_from_value_name(g, node.input[0])
        if input_shape is None or len(input_shape) != 5:
            continue
        weight_shape = helper.get_shape_from_value_name(g, node.input[1])
        if weight_shape is None:
            continue
        elif weight_shape[3] == 1 and weight_shape[4] == 1:
            node_to_remove.extend(defuse_Conv3D_to_Conv2D_kernel_k_1_1(g, node, input_shape))
        elif weight_shape[2] == 1:
            node_to_remove.extend(defuse_Conv3D_to_Conv2D_kernel_1_a_b(g, node, input_shape))
        else:
            continue
    for node in node_to_remove:
        g.node.remove(node)
    # Topological sort
    topological_sort(g)


def defuse_Conv3D_to_Conv2D_kernel_1_a_b(g, n, input_shape):
    output_shape = helper.get_shape_from_value_name(g, n.output[0])
    if output_shape is None:
        return []
    new_nodes = []
    # Check Conv attribute
    dilations = helper.get_list_attribute_by_name(n, 'dilations', 'int')
    if dilations is not None and dilations[0] != 1:
        return []
    elif dilations is None:
        new_dilations = [1, 1]
    else:
        new_dilations = dilations[1:]
    pads = helper.get_list_attribute_by_name(n, 'pads', 'int')
    if pads is not None and (pads[0] != 0 or pads[3] != 0):
        return []
    elif pads is None:
        new_pads = [0, 0, 0, 0]
    else:
        new_pads = pads[1:3] + pads[4:]
    strides = helper.get_list_attribute_by_name(n, 'strides', 'int')
    if strides is not None and strides[0] != 1:
        return []
    elif strides is None:
        new_strides = [1, 1]
    else:
        new_strides = strides[1:]
    group = helper.get_var_attribute_by_name(n, 'group', 'int')
    if group is None:
        group = 1
    elif group != 1:
        return []
    # Create Transpose, Reshape and Split nodes before
    transpose_perm = [0, 2, 1, 3, 4]
    transpose_prev = onnx.helper.make_node(
        'Transpose',
        [n.input[0]],
        [n.name + '_prev_transpose'],
        name=n.name + '_prev_transpose',
        perm=transpose_perm
    )
    new_input_shape = [input_shape[0] * input_shape[2], input_shape[1]] + input_shape[-2:]
    prev_shape_constant = helper.list_to_constant(
        n.name + '_prev_reshape_shape',
        [4],
        new_input_shape
    )
    reshape_prev = onnx.helper.make_node(
        'Reshape',
        [transpose_prev.output[0], prev_shape_constant.name],
        [n.name + '_prev_reshape'],
        name=n.name + '_prev_reshape'
    )
    split_prev = onnx.helper.make_node(
        'Split',
        [reshape_prev.output[0]],
        [n.name + f'_split_out_{i}' for i in range(new_input_shape[0])],
        name=n.name + '_split_out',
        axis=0,
        split=[1] * new_input_shape[0]
    )
    new_nodes.extend([prev_shape_constant, transpose_prev, reshape_prev, split_prev])
    # Create Conv nodes
    original_weight_node = helper.find_node_by_output_name(g, n.input[1])
    weight_shape, weight_value = helper.constant_to_list(original_weight_node)
    new_weight_node = helper.list_to_constant(
        n.input[1] + '_reshaped',
        weight_shape[:2] + weight_shape[3:],
        weight_value
    )
    new_weights = [new_weight_node.output[0]]
    new_nodes.append(new_weight_node)
    if len(n.input) > 2:
        new_weights.append(n.input[2])
    for i in range(new_input_shape[0]):
        new_conv_node = onnx.helper.make_node(
            'Conv',
            [n.name + f'_split_out_{i}'] + new_weights,
            [n.name + f'_split_conv_{i}'],
            name=n.name + f'_split_conv_{i}',
            dilations=new_dilations,
            pads=new_pads,
            strides=new_strides,
            group=group,
            kernel_shape=weight_shape[3:]
        )
        new_nodes.append(new_conv_node)
    # Create Concat, Reshape and Transpose nodes after
    concat_after = onnx.helper.make_node(
        'Concat',
        [n.name + f'_split_conv_{i}' for i in range(new_input_shape[0])],
        [n.name + '_concat'],
        name=n.name + '_concat',
        axis=0
    )
    new_nodes.append(concat_after)
    after_shape_constant = helper.list_to_constant(
        n.name + '_after_reshape_shape',
        [5],
        [output_shape[0], output_shape[2], output_shape[1]] + output_shape[3:]
    )
    reshape_after = onnx.helper.make_node(
        'Reshape',
        [concat_after.output[0], after_shape_constant.name],
        [n.name + '_after_reshape'],
        name=n.name + '_after_reshape'
    )
    new_nodes.append(after_shape_constant)
    new_nodes.append(reshape_after)
    transpose_after = onnx.helper.make_node(
        'Transpose',
        [reshape_after.output[0]],
        [n.output[0]],
        name=n.name + '_after_transpose',
        perm=[0, 2, 1, 3, 4]
    )
    new_nodes.append(transpose_after)
    # Modify the graph
    g.node.extend(new_nodes)
    modhelper.delete_value_with_name_if_exists(g, n.input[1])
    return [original_weight_node, n]


def defuse_Conv3D_to_Conv2D_kernel_k_1_1(g, n, input_shape):
    output_shape = helper.get_shape_from_value_name(g, n.output[0])
    if output_shape is None:
        return []
    new_nodes = []
    # Check Conv attribute
    dilations = helper.get_list_attribute_by_name(n, 'dilations', 'int')
    if dilations is not None and dilations[1:] != [1, 1]:
        return []
    elif dilations is None:
        new_dilations = [1, 1]
    else:
        new_dilations = dilations[:1]
    pads = helper.get_list_attribute_by_name(n, 'pads', 'int')
    if pads is not None and (pads[1:3] != [0, 0] or pads[-2:] != [0, 0]):
        return []
    elif pads is None:
        new_pads = [0, 0, 0, 0]
    else:
        new_pads = [pads[0], 0, pads[3], 0]
    strides = helper.get_list_attribute_by_name(n, 'strides', 'int')
    if strides is not None and strides[1:] != [1, 1]:
        return []
    elif strides is None:
        new_strides = [1, 1]
    else:
        new_strides = [strides[0], 1]
    group = helper.get_var_attribute_by_name(n, 'group', 'int')
    if group is None:
        group = 1
    # Create Reshape before
    new_input_shape = input_shape[:-2] + [input_shape[-2] * input_shape[-1]]
    new_shape_constant = helper.list_to_constant(
        n.name + '_prev_reshape_shape',
        [4],
        new_input_shape
    )
    reshape_prev = onnx.helper.make_node(
        'Reshape',
        [n.input[0], new_shape_constant.name],
        [n.name + '_prev_reshape'],
        name=n.name + '_prev_reshape'
    )
    new_nodes.append(new_shape_constant)
    new_nodes.append(reshape_prev)
    # Create new Conv
    original_weight_node = helper.find_node_by_output_name(g, n.input[1])
    weight_shape, weight_value = helper.constant_to_list(original_weight_node)
    new_weight_node = helper.list_to_constant(
        n.input[1] + '_reshaped',
        weight_shape[:-1],
        weight_value
    )
    new_inputs = [reshape_prev.output[0], new_weight_node.output[0]]
    if len(n.input) > 2:
        new_inputs.append(n.input[2])
    new_conv_node = onnx.helper.make_node(
        'Conv',
        new_inputs,
        [n.name + '_2d_conv'],
        name=n.name + '_2d_conv',
        dilations=new_dilations,
        pads=new_pads,
        strides=new_strides,
        group=group,
        kernel_shape=weight_shape[2:-1]
    )
    new_nodes.append(new_weight_node)
    new_nodes.append(new_conv_node)
    # Create Reshape after
    new_shape_constant = helper.list_to_constant(
        n.name + '_post_reshape_shape',
        [5],
        output_shape
    )
    reshape_post = onnx.helper.make_node(
        'Reshape',
        [new_conv_node.output[0], new_shape_constant.name],
        [n.output[0]],
        name=n.name + '_post_reshape'
    )
    new_nodes.append(new_shape_constant)
    new_nodes.append(reshape_post)
    # Modify the graph
    g.node.extend(new_nodes)
    modhelper.delete_value_with_name_if_exists(g, n.input[1])
    return [original_weight_node, n]
