import onnx
import onnx.helper
import argparse
import logging
import copy
import json
from collections import deque
from tools import eliminating, other, replacing, helper, constant_folding, modhelper

# For models with Loop node, we should carefully limit the use of polish_model.

debug = False
debug_path = ''

def onnx_shape_format(g):
    for input in g.input:
        for dim in input.type.tensor_type.shape.dim:
            if len(dim.dim_param) != 0:
                dim.dim_param = ''
                dim.dim_value = 1
    for output in g.output:
        for dim in output.type.tensor_type.shape.dim:
            if len(dim.dim_param) != 0:
                dim.dim_param = ''
                dim.dim_value = 1


def get_value_used_inside_loop(loop_node):
    """Get the value names used inside loop but not loop node input or inner results.

    Args:
        loop_node (NodeProto): the loop node.

    Returns:
        Set[str]: a set of names.
    """
    # Get loop node graph
    attr = helper.get_attribute_by_name(loop_node, 'body')
    loop_node_graph = attr.g
    # Add all loop defined values
    loop_defined_values = set()
    for value in loop_node_graph.input:
        loop_defined_values.add(value.name)
    for value in loop_node_graph.initializer:
        loop_defined_values.add(value.name)
    for node in loop_node_graph.node:
        for value_name in node.output:
            loop_defined_values.add(value_name)
    # Check if there is undefined values
    loop_undefined_values = set()
    for value in loop_node_graph.output:
        if value.name not in loop_defined_values and value.name not in loop_undefined_values:
            loop_undefined_values.add(value.name)
    for node in loop_node_graph.node:
        for value_name in node.input:
            if value_name not in loop_defined_values and value_name not in loop_undefined_values:
                loop_undefined_values.add(value_name)
    # Check for loop inside loop
    for node in loop_node_graph.node:
        if node.op_type == 'Loop':
            logging.error("Loop inside loop currently is not supported")
            exit(1)
    return loop_undefined_values


def all_shapes_are_inferenced(m):
    all_outputs = set()
    for node in m.graph.node:
        all_outputs = all_outputs.union(set(node.output))
    eliminating.eliminate_empty_value_infos(m.graph)
    available_outputs = set()
    available_outputs = available_outputs.union(set([value.name for value in m.graph.output]))
    available_outputs = available_outputs.union(set([value.name for value in m.graph.value_info]))
    return not (len(available_outputs) < len(all_outputs))


def get_loop_nodes_inputs_values(model, loop_input_names):
    """Get the input value_info for all loop inputs, including None standard inputs.

    Args:
        model (ModelProto): the model
        loop_node (NodeProto): the loop node to process
    Returns:
        Dict|None: the input values dictionary if all of them are ready. Else return None.
    """
    loop_inputs = dict()
    for name in loop_input_names:
        value = helper.find_value_by_name(model.graph, name)
        if value is None:
            value = helper.find_input_by_name(model.graph, name)
        if value is None:
            return None
        else:
            loop_inputs[name] = value
    return loop_inputs


def remove_following_shapes(g, value_info_name):
    todo = deque()
    todo.extend(helper.find_following_nodes_by_input_value_name(g, value_info_name))
    while len(todo) != 0:
        node = todo.popleft()
        for output in node.output:
            value = helper.find_value_by_name(g, output)
            if value is not None:
                g.value_info.remove(value)
            todo.extend(helper.find_following_nodes_by_input_value_name(g, output))


def get_max_loop_count(g, loop_node):
    """For loop node working with less, just need to get a number other than 0.

    Args:
        g (GraphProto): the outer graph
        loop_node (NodeProto): the loop node

    Returns:
        int: the max loop count
    """
    # Try to check loop node comment
    if len(loop_node.doc_string) != 0:
        try:
            doc_json = json.loads(loop_node.doc_string)
        except ValueError:
            doc_json = {'doc_string': loop_node.doc_string}
        if 'max_loop_count' in doc_json:
            return doc_json[max_loop_count]
    # Inference the value
    m_name = loop_node.input[0]
    node = helper.find_node_by_output_name(g, m_name)
    if node is not None:
        max_loop_count = helper.constant_to_numpy(node)[0]
    else:
        init = helper.find_initializer_by_name(g, m_name)
        if init is None:
            max_loop_count = -1
        else:
            max_loop_count = helper.initializer_to_numpy(init)[0]
    for i in range(2, len(loop_node.input)):
        input_name = loop_node.input[i]
        node = helper.find_node_by_output_name(g, input_name)
        if node is None:
            continue
        if node.op_type != 'Constant':
            continue
        shape, value = helper.constant_to_list(node)
        if shape != 1:
            continue
        elif value[0] == 0:
            continue
        elif max_loop_count > 0 and max_loop_count <= 65535:
            logging.error(f"Multiple int inputs for {loop_node.name}")
            exit(1)
        else:
            max_loop_count = value[0]
    if max_loop_count > 65535 or max_loop_count < 1:
        logging.error(f"Cannot inference loop count for {loop_node.name}.")
        exit(1)
    # Save the max_loop_count into doc_string
    new_loop = helper.find_node_by_node_name(g, loop_node.name)
    if len(loop_node.doc_string) == 0:
        new_loop.doc_string = json.dumps({'max_loop_count': max_loop_count})
    else:
        doc_json['max_loop_count'] = max_loop_count
        new_loop.doc_string = json.dumps(doc_json)
    return max_loop_count


def loop_node_process(m, loop_node, input_values):
    """Shape inference inside the loop.

    Args:
        m (ModelProto): the outer model.
        loop_node (NodeProto): the loop node.
        input_values (Dict): input name to input value mapping.

    Returns:
        ModelProto: the processed model.
    """
    # 1. Setup input and output. Add hidden inputs
    inner_graph = loop_node.attribute[0].g
    # 2, Create temporary model.
    for name in input_values:
        # Hidden inputs should be added.
        if name not in loop_node.input:
            loop_node.input.append(name)
            loop_node.output.insert(len(loop_node.input) - 3, name + '_matching_output')
            inner_graph.input.append(input_values[name])
            inner_graph.output.insert(len(loop_node.input) - 2 ,input_values[name])
    temp_model = onnx.helper.make_model(inner_graph)
    temp_model.opset_import[0].version = 12
    temp_model.ir_version = 7
    # 3. Shape inference.
    temp_model = other.inference_shapes(temp_model)
    replacing.replace_split_with_slices(temp_model.graph)
    other.topological_sort(temp_model.graph)
    temp_model = other.inference_shapes(temp_model)
    if debug:
        onnx.save(temp_model, f"{debug_path}/debug_inner_{loop_node.name.replace('/', '_')}.onnx")
    # 4. Extract loop to outer.
    new_inner_graph = temp_model.graph
    loop_node.attribute.pop()
    loop_node.attribute.append(onnx.helper.make_attribute("body", value=new_inner_graph))
    # 5. Redo outer shape inference
    m = other.inference_shapes(m)
    carried_var_num = len(loop_node.input) - 2
    # first n outputs
    for i in range(0, carried_var_num):
        output_value = helper.find_value_by_name(m.graph, loop_node.output[i])
        if output_value is not None and len(output_value.type.tensor_type.shape.dim) == 0:
            m.graph.value_info.remove(output_value)
        elif output_value is not None:
            continue
        new_value = copy.deepcopy(new_inner_graph.output[i + 1])
        new_value.name = loop_node.output[i]
        m.graph.value_info.append(new_value)
    # following k outputs
    max_loop_count = get_max_loop_count(m.graph, loop_node)
    for i in range(carried_var_num, len(loop_node.output)):
        output_value = helper.find_value_by_name(m.graph, loop_node.output[i])
        if output_value is None:
            output_value = helper.find_output_by_name(m.graph, loop_node.output[i])
        if output_value is None:
            logging.warn(f"Cannot get the output shape of Loop: {loop_node.output[i]}")
            continue
        output_value.type.tensor_type.shape.dim[0].dim_value = max_loop_count
        remove_following_shapes(m.graph, loop_node.output[i])
    m = other.inference_shapes(m)
    return m


def eliminate_no_usage_constant(g, loop_node_inputs):
    used_name_set = set()
    for node in g.node:
        for input in node.input:
            used_name_set.add(input)
    for output in g.output:
        used_name_set.add(output.name)
    for name in loop_node_inputs:
        used_name_set.add(name)
    for node in g.node:
        if node.op_type == 'Constant' and node.name not in used_name_set:
            g.node.remove(node)


def compiler_onnx_process(m):
    modhelper.setup_current_opset_version(m)
    # Format shapes in the first place
    onnx_shape_format(m.graph)
    m = other.inference_shapes(m)
    other.add_name_to_node(m.graph)
    other.rename_all_node_name(m.graph)
    # Find loop node
    todo_loop_names = deque()
    loop_input_name_mapping = dict()
    loop_inside_usage_values = set()
    for node in m.graph.node:
        if node.op_type == 'Loop':
            # Add loop node to todo list
            todo_loop_names.append(node.name)
            loop_node_input_names = set([name for name in node.input])
            inner_usege = get_value_used_inside_loop(node)
            loop_node_input_names = loop_node_input_names.union(inner_usege)
            loop_inside_usage_values = loop_inside_usage_values.union(inner_usege)
            loop_input_name_mapping[node.name] = loop_node_input_names
            # Format loop node shape if any
            loop_graph = helper.get_attribute_by_name(node, 'body').g
            onnx_shape_format(loop_graph)
    # Check if all values are inferred.
    # If all the required information are there, return the model. This is a very simple model.
    if len(todo_loop_names) == 0 and all_shapes_are_inferenced(m):
        logging.info("All shapes are inferenced. Done.")
        return m
    # If not, use while loop to deal with node one by one
    loop_time = 0
    last_value_info_count = len(m.graph.value_info)
    last_todo_loop_count = len(todo_loop_names)
    last_node_count = len(m.graph.node)
    replacing.replace_initializer_with_Constant(m.graph, False)
    if debug:
        onnx.save(m, f"{debug_path}/debug_0.onnx")
    while len(todo_loop_names) != 0 or not all_shapes_are_inferenced(m):
        eliminating.eliminate_empty_value_infos(m.graph)
        # Try constant folding.
        replacing.replace_shape_with_constant(m.graph)
        replacing.replace_ConstantOfShape_with_constant(m.graph)
        m = other.inference_shapes(m)
        while constant_folding.constant_folding(m.graph):
            other.topological_sort(m.graph)
            while len(m.graph.value_info) != 0:
                m.graph.value_info.pop()
        m = other.inference_shapes(m)
        replacing.replace_shape_with_constant(m.graph)
        other.topological_sort(m.graph)
        m = other.inference_shapes(m)
        # Try solving loop nodes.
        if len(todo_loop_names) != 0:
            todo_loop_name = todo_loop_names[0]
            todo_loop = helper.find_node_by_node_name(m.graph, todo_loop_name)
            input_values = get_loop_nodes_inputs_values(m, loop_input_name_mapping[todo_loop_name])
            if input_values is not None:
                logging.debug(f"Processing loop node {todo_loop_names}")
                m = loop_node_process(m, todo_loop, input_values)
                todo_loop_names.popleft()
        # Normally inference shapes last
        eliminate_no_usage_constant(m.graph, loop_inside_usage_values)
        m = other.inference_shapes(m)
        # Some statistics
        loop_time += 1
        logging.debug(f"Process {loop_time} times.")
        if debug:
            onnx.save(m, f"{debug_path}/debug_{loop_time}.onnx")
        if len(m.graph.value_info) == last_value_info_count and \
            len(todo_loop_names) == last_todo_loop_count and \
            len(m.graph.node) == last_node_count:
            # Nothing happens, stop and debug.
            logging.warning(f"Model process dead loop. Total loop times: {loop_time}")
            return m
        else:
            last_value_info_count = len(m.graph.value_info)
            last_todo_loop_count = len(todo_loop_names)
            last_node_count = len(m.graph.node)
    # Normally ended
    logging.info("Model successfully processed.")
    return m


# Test process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Help converter deal with models without shapes and models with Loop.')
    parser.add_argument('in_file', help='input file')
    parser.add_argument('out_file', help='output optimized model file')
    parser.add_argument('--debug', action='store_true', default=False, required=False)
    parser.add_argument('--debugpath', type=str, required=False)

    args = parser.parse_args()
    if args.debug:
        debug = True
        debug_path = args.debugpath if args.debugpath else ''
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    m = onnx.load(args.in_file)

    m = compiler_onnx_process(m)

    onnx.save(m, args.out_file)