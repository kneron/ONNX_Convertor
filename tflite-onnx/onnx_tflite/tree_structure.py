import tensorflow as tf

from conv_layers import Convolution, DepthwiseConvolution, ResizeNearestNeighbor, ResizeBilinear, TransposeConvolution
from aact_layers import Relu, Relu6, Softmax, LOGISTIC, PRelu
from core_layers import Dense, Reshape, Pad, Squeeze, L2Normalization
from merg_layers import Add, Mul, Concatenation
from pool_layers import MaxPooling2D, AveragePooling2D, Mean

from tflite.BuiltinOperator import BuiltinOperator
from tflite.Model import Model

# For Testing and Check Graph Visualization
def make_graph(tree):
    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot, graphviz_layout
    import matplotlib.pyplot as plt

    graph = nx.DiGraph(directed=True)
    nodes = tree.get_nodes()

    for node_name in nodes:
        graph.add_node(node_name)

        for output_node_name in nodes[node_name].output_nodes_name:
            graph.add_edge(node_name, output_node_name)

    pos = graphviz_layout(graph, prog='dot')
    nx.draw(graph, with_labels=True, pos=pos, font_weight='bold')
    plt.show()

# Core Tree Structure Class
class Tree:
    def __init__(self, model_path, defused=True):
        # parse operator information through flatc python module
        self.__init_op_info(model_path)

        # parse node information through tflite interpreter (tflite interpreter can't parse operator information in our target tensorflow version 1.15)
        self.__interpreter = tf.lite.Interpreter(model_path)
        self.__interpreter.allocate_tensors()

        self.__parse_graph()
        self.__eliminate_side_input()
        self.__init_inputs_node_info()
        self.__init_outputs_node_info()
        self.__defused(enable_defuse=defused)
        self.__init_graph_inputs_node()
        self.__init_graph_outputs_node()

    def __init_op_info(self, model_path):
        self.__tflite_ops = []
        self.__tflite_op_types = []

        data = open(model_path, "rb").read()
        raw_model = Model.GetRootAsModel(bytearray(data), 0)

        tflite_graph = raw_model.Subgraphs(0)

        for idx in range(tflite_graph.OperatorsLength()):
            op = tflite_graph.Operators(idx)
            op_type = raw_model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

            self.__tflite_ops.append(op)
            self.__tflite_op_types.append(op_type)

    def __parse_graph(self):
        self.__nodes = dict()
        self.__sequential_nodes_key = []

        for idx, op in enumerate(self.__tflite_ops):
            node = self.__node_generator(op=op,
                                         op_type=self.__tflite_op_types[idx],
                                         tflite_interpreter=self.__interpreter)
            self.__nodes[node.node_name] = node
            self.__sequential_nodes_key.append(node.node_name)

    def __eliminate_side_input(self):
        # eliminate side input idx
        nodes_idx = [self.__nodes[node_name].node_idx for node_name in self.__nodes]
        for node_name in self.__nodes:
            new_input_nodes_idx = []
            for input_node_idx in self.__nodes[node_name].input_nodes_idx:
                if input_node_idx in nodes_idx:
                    new_input_nodes_idx.append(input_node_idx)
            self.__nodes[node_name].input_nodes_idx = new_input_nodes_idx

        # eliminate side input name
        nodes_name = self.__nodes.keys()
        for node_name in self.__nodes:
            new_input_nodes_name = []
            for input_node_name in self.__nodes[node_name].input_nodes_name:
                if input_node_name in nodes_name:
                    new_input_nodes_name.append(input_node_name)
            self.__nodes[node_name].input_nodes_name = new_input_nodes_name

    def __init_inputs_node_info(self):
        if self.__nodes is None:
            raise ValueError("__nodes not initial")

        # init input_nodes
        for node_name in self.__nodes:
            node = self.__nodes[node_name]
            for input_node_name in node.input_nodes_name:
                node.input_nodes.append(self.__nodes[input_node_name])

    def __init_outputs_node_info(self):
        if self.__nodes is None:
            raise ValueError("__nodes not initial")

        for node_name in self.__nodes:
            node = self.__nodes[node_name]
            node.output_nodes_idx.clear()
            node.output_nodes_name.clear()
            node.output_nodes.clear()

        # init output_nodes_idx
        nodes_idx_dict = {}
        for node_name in self.__nodes:
            nodes_idx_dict[self.__nodes[node_name].node_idx] = self.__nodes[node_name]

        for idx in nodes_idx_dict:
            node = nodes_idx_dict[idx]
            for input_node_idx in node.input_nodes_idx:
                if not(idx in nodes_idx_dict[input_node_idx].output_nodes_idx):
                    nodes_idx_dict[input_node_idx].output_nodes_idx.append(idx)

        for idx in nodes_idx_dict:
            self.__nodes[nodes_idx_dict[idx].node_name].output_nodes_idx = nodes_idx_dict[idx].output_nodes_idx

        # init output_nodes_name
        for node_name in self.__nodes:
            node = self.__nodes[node_name]
            for input_node_name in node.input_nodes_name:
                if not (node_name in self.__nodes[input_node_name].output_nodes_name):
                    self.__nodes[input_node_name].output_nodes_name.append(node_name)

        # init output_nodes
        for node_name in self.__nodes:
            node = self.__nodes[node_name]
            for input_node in node.input_nodes:
                if not (node in input_node.output_nodes):
                    input_node.output_nodes.append(node)

    def __defused(self, enable_defuse):
        if not enable_defuse:
            return

        add_defused_activation_node_list = []
        for node_name in self.__nodes:
            node = self.__nodes[node_name]

            defused_activation_node = node.defuse_activation_function()

            # add new fused activation node by `node name` link
            if None is not defused_activation_node:
                defused_activation_node.node_idx = -1

                # init input node
                input_node_outputs_remove_name = node.output_nodes_name.copy()
                input_node_outputs_add_name = defused_activation_node.node_name

                input_node_outputs_remove_node = node.output_nodes.copy()
                input_node_outputs_add_node = defused_activation_node

                # init fused node
                fused_node_inputs_name = [node.node_name]
                fused_node_outputs_name = node.output_nodes_name.copy()

                fused_node_inputs_node = [node]
                fused_node_outputs_node = node.output_nodes.copy()

                # init output node
                output_node_inputs_remove_node_name = node.node_name
                output_node_inputs_add_node_name = defused_activation_node.node_name

                output_node_inputs_remove_node = node
                output_node_inputs_add_node = defused_activation_node

                # modify input node
                for input_node_output_remove_name in input_node_outputs_remove_name:
                    node.output_nodes_name.remove(input_node_output_remove_name)
                node.output_nodes_name.append(input_node_outputs_add_name)

                for input_node_output_remove_node in input_node_outputs_remove_node:
                    node.output_nodes.remove(input_node_output_remove_node)
                node.output_nodes.append(input_node_outputs_add_node)

                # modify fused node
                defused_activation_node.input_nodes_name = fused_node_inputs_name
                defused_activation_node.output_nodes_name = fused_node_outputs_name

                defused_activation_node.input_nodes = fused_node_inputs_node
                defused_activation_node.output_nodes = fused_node_outputs_node

                # modify output node
                for output_node_name in fused_node_outputs_name:
                    output_node = self.__nodes[output_node_name]
                    output_node.input_nodes_name.remove(output_node_inputs_remove_node_name)
                    output_node.input_nodes_name.append(output_node_inputs_add_node_name)

                for output_node in fused_node_outputs_node:
                    output_node.input_nodes.remove(output_node_inputs_remove_node)
                    output_node.input_nodes.append(output_node_inputs_add_node)

                # defused node is not head node
                defused_activation_node.is_head_node = False

                # origin node is not buttom node
                defused_activation_node.is_bottom_node = node.is_bottom_node
                node.is_bottom_node = False

                # add fused node
                add_defused_activation_node_list.append(defused_activation_node)

        # add fused node
        for defused_activation_node in add_defused_activation_node_list:
            self.__nodes[defused_activation_node.node_name] = defused_activation_node

            # update __sequential_nodes_key
            insert_index = self.__sequential_nodes_key.index(defused_activation_node.input_nodes[0].node_name) + 1
            self.__sequential_nodes_key.insert(insert_index, defused_activation_node.node_name)

    def __init_graph_inputs_node(self):
        self.__head_nodes = []

        for node_name in self.__nodes:
            if True is self.__nodes[node_name].is_head_node:
                self.__head_nodes.append(self.__nodes[node_name])

    def __init_graph_outputs_node(self):
        self.__bottom_nodes = []

        for node_name in self.__nodes:
            if True is self.__nodes[node_name].is_bottom_node:
                self.__bottom_nodes.append(self.__nodes[node_name])


    def __node_generator(self, op, op_type, tflite_interpreter):
        if op_type == BuiltinOperator.CONV_2D:
            layer_obj = Convolution(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.DEPTHWISE_CONV_2D:
            layer_obj = DepthwiseConvolution(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.SOFTMAX:
            layer_obj = Softmax(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.RELU:
            layer_obj = Relu(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.RELU6:
            layer_obj = Relu6(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.PRELU:
            layer_obj = PRelu(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.LOGISTIC:
            layer_obj = LOGISTIC(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.FULLY_CONNECTED:
            layer_obj = Dense(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.RESHAPE:
            layer_obj = Reshape(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.PAD:
            layer_obj = Pad(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.ADD:
            layer_obj = Add(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.MUL:
            layer_obj = Mul(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.CONCATENATION:
            layer_obj = Concatenation(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.MEAN:
            layer_obj = Mean(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.MAX_POOL_2D:
            layer_obj = MaxPooling2D(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.AVERAGE_POOL_2D:
            layer_obj = AveragePooling2D(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.SQUEEZE:
            layer_obj = Squeeze(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
            layer_obj = ResizeNearestNeighbor(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.RESIZE_BILINEAR:
            layer_obj = ResizeBilinear(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.L2_NORMALIZATION:
            layer_obj = L2Normalization(op, op_type, tflite_interpreter)
        elif op_type == BuiltinOperator.TRANSPOSE_CONV:
            layer_obj = TransposeConvolution(op, op_type, tflite_interpreter)
        else:
            raise ValueError(op_type)

        return layer_obj

    def get_head_nodes(self):
        return self.__head_nodes

    def get_bottom_nodes(self):
        return self.__bottom_nodes

    def get_sequential_nodes_key(self):
        return self.__sequential_nodes_key.copy()

    def get_nodes(self):
        return self.__nodes

## Example:
####################
# tree_graph = Tree(
#     model_path='/home/andy_huang/data/tf_detection_model_zoo/coco_trained_models/ssd_inception_v2_coco/ssd_inception_v2_coco_2018_01_28/saved_model_ssd/model.tflite',
#     defused=True
#     )
# # print(tree_graph.get_nodes())
# make_graph(tree_graph)
#
# node_list = tree_graph.get_head_nodes()
#
# for node in node_list:
#     output_nodes = node.output_nodes
#     print(node.node_name)
#     while 0 < output_nodes.__len__():
#         print(output_nodes[0].node_name)
#         output_nodes = output_nodes[0].output_nodes
