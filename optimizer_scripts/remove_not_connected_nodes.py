from collections import deque
import logging
from tools import other, replacing, helper
import sys
import onnx

def get_set_of_connected_node_names(g):
    # Input to output approach
    connected_node_names = set()
    top_names = set([v.name for v in g.input])
    to_visit = deque()
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
    to_visit = deque([n for n in g.node if n.name in connected_node_names])


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


m = onnx.load(sys.argv[1])
replacing.replace_initializer_with_Constant(m.graph)
eliminate_not_connected_nodes(m.graph)
other.topological_sort(m.graph)
onnx.save(m, sys.argv[2])