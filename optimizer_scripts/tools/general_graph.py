from collections import deque

class Node:
    """A Node which maps a node proto. It has pointers to its parents and
    children.
    """
    def __init__(self, onnx_node):
        """Initialize a node. This initialization only set up the mapping to
        node proto. The pointers should be set up by outside.
        """
        self.name = None
        self.parents = []
        self.children = []
        self.proto = None
        self.output_value = None
        if onnx_node is not None:
            self.name = onnx_node.name
            self.proto = onnx_node

class Graph:
    """A graph which is constructed from the onnx proto.
    """
    def __init__(self, onnx_graph):
        """Construct the graph from onnx.
        """
        self.input_nodes = []
        self.output_nodes = []
        self.name2node = {}
        self.output2node = {}
        self.proto = onnx_graph
        # Add input nodes
        for value in onnx_graph.input:
            input_node = Node(None)
            input_node.name = "Input_" + value.name
            input_node.output_value = value
            self.name2node[input_node.name] = input_node
            self.output2node[value.name] = input_node
            self.input_nodes.append(input_node)
        output_value_names = [value.name for value in onnx_graph.output]
        # Add regular nodes
        for onnx_node in onnx_graph.node:
            node = Node(onnx_node)
            self.name2node[node.name] = node
            self.output2node[onnx_node.output[0]] = node
            for value_name in onnx_node.input:
                node.parents.append(self.output2node[value_name])
                self.output2node[value_name].children.append(node)
            if onnx_node.output[0] in output_value_names:
                self.output_nodes.append(node)
        # Add value infos
        for value in onnx_graph.value_info:
            node = self.output2node[value.name]
            node.output_value = value
    def get_sorted_node_list(self):
        """Return a node list in topological order.
        """
        visited = set()
        todo = deque()
        result = []
        for node in self.input_nodes:
            todo.append(node)
            visited.add(node)
        for onnx_node in self.proto.node:
            if onnx_node.op_type == "Constant":
                node = self.name2node[onnx_node.name]
                todo.append(node)
                visited.add(node)
        while todo:
            node = todo.popleft()
            result.append(node)
            for child in node.children:
                if child in visited:
                    continue
                ready = True
                for child_parent in child.parents:
                    if child_parent in visited:
                        continue
                    ready = False
                    break
                if ready:
                    todo.append(child)
                    visited.add(child)
        return result
