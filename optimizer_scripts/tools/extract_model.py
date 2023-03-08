import onnx
import onnx.helper
from collections import deque
from . import helper, other

def create_value_info(name, type, shape):
    elem_type = 0
    if type == 'float':
        elem_type = 1    # FLOAT
    elif type == 'int':
        elem_type = 7    # INT64
    elif type == 'bool':
        elem_type = 9
    else:
        helper.logger.error(f"Not supported data type: {type}")
        exit(1)

    new_value = onnx.helper.make_tensor_value_info(
        name,
        elem_type,
        shape)
    return new_value


def remove_not_connected_nodes(g):
    # Remove nodes that are not reachable by output node. BFS
    last_nodes = []
    for output in g.output:
        node = helper.find_node_by_output_name(g, output.name)
        if node not in last_nodes:
            last_nodes.append(node)
    visited = set([node.output[0] for node in last_nodes])
    to_visit = deque(last_nodes)
    while len(to_visit) > 0:
        visiting = to_visit.popleft()
        for input in visiting.input:
            if input in visited:
                continue
            input_node = helper.find_node_by_output_name(g, input)
            if input_node is None:
                continue
            else:
                to_visit.append(input_node)
                visited.add(input)
    to_remove = []
    for node in g.node:
        if node.output[0] not in visited:
            to_remove.append(node)
    for node in to_remove:
        g.node.remove(node)
    # Remove nodes that are not reachable by input node.
    other.topological_sort(g)

def extract_model(model_in, inputs=None, outputs=None):
    """Extract the subgraph to a model.

    Args:
        model_in (onnx.ModelProto): The input onnx model
        inputs (List): The inputs of the subgraph. A List of tuples of name, type and shape, e.g. [('input_0', 'float', (1, 3, 224, 224))]
        outputs (List): The outputs of the subgraph. A List of tuples of name, type and shape, e.g. [('output_0', 'float', (1, 3))]

    Returns:
        onnx.ModelProto: The subgraph model
    """
    # Create new input and outputs
    g = model_in.graph
    if inputs is not None:
        while len(g.input) > 0:
            g.input.pop()
        for input in inputs:
            input_value = create_value_info(input[0], input[1], input[2])
            g.input.append(input_value)
    if outputs is not None:
        while len(g.output) > 0:
            g.output.pop()
        for output in outputs:
            output_value = create_value_info(output[0], output[1], output[2])
            g.output.append(output_value)
    # Remove not connected nodes
    remove_not_connected_nodes(g)
    # Remove value_infos
    while len(g.value_info) > 0:
        g.value_info.pop()
    return model_in