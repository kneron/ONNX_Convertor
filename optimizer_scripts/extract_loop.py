import logging
import numpy as np
import sys
import onnx
from tools import other, replacing, helper, modhelper
from onnx import optimizer
from copy import deepcopy

def get_carried_var_name(g, pos):
    return (g.input[pos + 2].name, g.output[pos + 1].name)

def rename_in_loop_item(count, original_name):
    if count == -1:
        return original_name
    return "loop_{}_{}".format(count, original_name)

def multiply_Loop_graph_and_replace(g, loop_node, loop_times):
    inner_graph = loop_node.attribute[0].g
    inner_graph_input_names = set([input.name for input in inner_graph.input])
    # Check all the input and output to find the pass through variables
    total_carried_var_count = len(loop_node.input) - 2
    go_through_inputs = set()
    for i in range(total_carried_var_count):
        in_name, out_name = get_carried_var_name(inner_graph, i)
        if in_name == out_name:
            go_through_inputs.add(in_name)
    # Note all inner generated names, which name should be changed every iteration
    inner_generated_var_names = set()
    for n in inner_graph.node:
        for output_name in n.output:
            inner_generated_var_names.add(output_name)
    for init_var in inner_graph.initializer:
        inner_generated_var_names.add(init_var.name)

    # Multiply and rename all the nodes
    generated_nodes = []
    for i in range(loop_times):
        for n in inner_graph.node:
            copied_node = deepcopy(n)
            copied_node.name = rename_in_loop_item(i, n.name)

            # Rename inputs
            reversed_input_names = []
            while len(copied_node.input) > 0:
                reversed_input_names.append(copied_node.input.pop())
            # There are many situations:
            while len(reversed_input_names) != 0:
                original_name = reversed_input_names.pop()
                # 1. The input is a go-through input. We keep it unchanged.
                if original_name in go_through_inputs:
                    copied_node.input.append(original_name)
                # 2. The input is an inner graph input.
                elif original_name in inner_graph_input_names:
                    # For the first loop, the inner input is the outer input
                    if i == 0:
                        copied_node.input.append(original_name)
                        continue
                    # For the following loops, the rename should based on last loop count and the output name.
                    for j in range(total_carried_var_count):
                        in_name, out_name = get_carried_var_name(inner_graph, j)
                        if in_name == original_name:
                            copied_node.input.append(rename_in_loop_item(i-1, out_name))
                            break
                # 3. The input is a loop generated variable. The rename should based on current loop count.
                elif original_name in inner_generated_var_names:
                    copied_node.input.append(rename_in_loop_item(i, original_name))
                # 5. The input is an outer variable. We keep it unchanged.
                else:
                    copied_node.input.append(original_name)

            # Rename output
            reversed_output_names = []
            while len(copied_node.output) != 0:
                reversed_output_names.append(copied_node.output.pop())
            while len(reversed_output_names) != 0:
                reversed_output_name = reversed_output_names.pop()
                copied_node.output.append(rename_in_loop_item(i, reversed_output_name))

            # Add the node
            generated_nodes.append(copied_node)
    # Concat the scan_outputs
    for i in range(len(loop_node.output) - total_carried_var_count):
        outer_output_name = loop_node.output[total_carried_var_count + i]
        input_names = []
        # For each loop, introduce a Unsqueeze.
        for j in range(loop_times):
            the_scan_output_name = rename_in_loop_item(j, inner_graph.output[total_carried_var_count + 1 + i].name)
            unsqueeze_node_name = "unsqueeze_" + the_scan_output_name
            # Create the Unsqueeze
            unsqueeze_node = onnx.helper.make_node(
                "Unsqueeze",
                [the_scan_output_name],
                [unsqueeze_node_name],
                name=unsqueeze_node_name,
                axes=[0]
            )
            generated_nodes.append(unsqueeze_node)
            # Append the output to input names
            input_names.append(unsqueeze_node_name)
        # Create the concat node
        concat_node = onnx.helper.make_node(
            "Concat",
            input_names,
            [outer_output_name],
            name=outer_output_name,
            axis = 0
        )
        generated_nodes.append(concat_node)

    # Multiply and rename initializers
    generated_initializers = []
    for i in range(loop_times):
        for init_var in inner_graph.initializer:
            copied_init_var = deepcopy(init_var)
            copied_init_var.name = rename_in_loop_item(i, init_var.name)
            generated_initializers.append(copied_init_var)

    # replace the inner graph input name to the outer graph tensor name
    for i in range(len(loop_node.input)):
        inner_input_name = inner_graph.input[i].name
        outer_input_name = loop_node.input[i]
        for n in generated_nodes:
            if inner_input_name in n.input:
                modhelper.replace_node_input(n, inner_input_name, outer_input_name)

    # replace the following output connected to loop carried values if any.
    for i in range(total_carried_var_count):
        the_carried_var_name = loop_node.output[i]
        the_inner_output_name = inner_graph.output[i + 1].name
        for n in g.node:
            if the_carried_var_name in n.input:
                if the_inner_output_name in go_through_inputs:
                    modhelper.replace_node_input(n, the_carried_var_name, the_inner_output_name)
                else:
                    modhelper.replace_node_input(n, the_carried_var_name, rename_in_loop_item(loop_times - 1, the_inner_output_name))

    # Delete loop node and extend new nodes.
    g.node.remove(loop_node)
    g.node.extend(generated_nodes)
    g.initializer.extend(generated_initializers)

def check_loop_times(g, loop_node):
    # Find loop node's M
    m_v = helper.weight_to_numpy(g, loop_node.input[0])
    if m_v is None:
        return None
    m_v = m_v[0]
    # Find loop node's first condition, should be true
    cond_node = helper.find_node_by_output_name(g, loop_node.input[1])
    if cond_node.op_type != 'Less':
        return None
    a = helper.weight_to_numpy(g, cond_node.input[0])
    b = helper.weight_to_numpy(g, cond_node.input[1])
    if a is None or b is None:
        return None
    if a[0] >= b[0]:
        return 0
    # Check inner condition calcuation. Only support add.
    inner_graph = loop_node.attribute[0].g
    # 1. Get the first output, which is the condition
    cond_output = inner_graph.output[0]
    # 2. Find previous node until find check node. Currently, we only support Less.
    ignore_ops = set(["Cast", "Identity"])
    cond_node = helper.find_node_by_output_name(inner_graph, inner_graph.output[0].name)
    while cond_node is not None and cond_node.op_type in ignore_ops:
        cond_node = helper.find_node_by_output_name(inner_graph, cond_node.input[0].name)
    if cond_node is None:
        return None
    if cond_node.op_type != 'Less':
        return None
    # 3. Find the input of the limit.
    the_name = cond_node.input[1]
    input_value = helper.find_input_by_name(inner_graph, the_name)
    while input_value is None:
        previous_node = helper.find_node_by_output_name(inner_graph, the_name)
        if previous_node is None:
            return None
        elif previous_node.op_type not in ignore_ops:
            return None
        the_name = previous_node.input[0]
        input_value = helper.find_input_by_name(inner_graph, the_name)
    index = -1
    for i in range(len(inner_graph.input)):
        if inner_graph.input[i].name != the_name:
            continue
        index = i
        break
    limit = helper.weight_to_numpy(g, loop_node.input[index])
    # limit should not change
    in_name, out_name = get_carried_var_name(inner_graph, index - 2)
    if in_name != out_name:
        return None
    if limit is None:
        return None
    limit = limit[0]
    # 4. Find the Add node
    the_name = cond_node.input[0]
    add_node = helper.find_node_by_output_name(inner_graph, the_name)
    while add_node is not None and add_node.op_type in ignore_ops:
        add_node = helper.find_node_by_output_name(inner_graph, add_node.input[0])
    if add_node is None:
        return None
    if add_node.op_type != 'Add':
        return None
    step = helper.weight_to_numpy(inner_graph, add_node.input[1])
    if step is None:
        return None
    step = step[0]
    # 5. Find the input of the init value.
    the_name = add_node.input[0]
    input_value = helper.find_input_by_name(inner_graph, the_name)
    while input_value is None:
        previous_node = helper.find_node_by_output_name(inner_graph, the_name)
        if previous_node is None:
            return None
        elif previous_node.op_type not in ignore_ops:
            return None
        the_name = previous_node.input[0]
        input_value = helper.find_input_by_name(inner_graph, the_name)
    index = -1
    for i in range(len(inner_graph.input)):
        if inner_graph.input[i].name != the_name:
            continue
        index = i
        break
    init_value = helper.weight_to_numpy(g, loop_node.input[index])
    if init_value is None:
        return None
    init_value = init_value[0]
    # add_node should output into the carried on value
    if add_node.output[0] != get_carried_var_name(inner_graph, index - 2)[1]:
        return None
    # Calculate count
    count = 0
    i = init_value
    while i < limit:
        i += step
        count += 1
    return count

def extract_Loop(m):
    g = m.graph
    for node in g.node:
        # Find a loop node
        if node.op_type != 'Loop':
            continue
        # Check iteration times
        iter_times = check_loop_times(g, node)
        if iter_times is None or iter_times < 0:
            logging.error("Cannot get the iteration count for Loop node: " + node.name)
            exit(1)
        multiply_Loop_graph_and_replace(g, node, iter_times)
    # Final cleaning
    replacing.replace_initializer_with_Constant(m.graph)
    other.topological_sort(m.graph)
    m = optimizer.optimize(m, ['eliminate_deadend'])
    return m

if __name__ == '__main__':
    m = onnx.load(sys.argv[1])
    m = extract_Loop(m)

    onnx.save(m, sys.argv[2])