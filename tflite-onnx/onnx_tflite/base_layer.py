"""Base layer class
"""
import abc
import logging


class Layer(metaclass=abc.ABCMeta):

    def __init__(self, previous_onnx_node_names, op_type, op_info, tflite_interpreter):
        self.op_type = op_type
        self.op_info = op_info
        self.tflite_interpreter = tflite_interpreter

        self.logger = logging.getLogger("onnx-tflite")
        self.logger.setLevel(logging.DEBUG)

        self.node_output_detail = tflite_interpreter._get_tensor_details(self.op_info['outputs'][0])
        self.node_input_detail = tflite_interpreter._get_tensor_details(self.op_info['inputs'][0])

        self.onnx_node_name = self.node_output_detail['name']
        self.previous_onnx_node_names = previous_onnx_node_names

        self.node_list = []
        self.value_infos = []
        self.weight_node_list = []

    @abc.abstractmethod
    def generate(self):
        """Generate the nodes according to the original layer

         Returns
        -------
        node_list : list
            a list of nodes generated
        value_infos : list
            value_infos between nodes
        """
        return NotImplemented
