import onnx
import onnx.helper
import logging
import sys

from tools import helper, other, replacing
from opset_9_to_11 import replace_min_max_attribute_to_const_node_in_clip_node, replace_all_attribute_to_const_node_in_pad_node, upsampling_to_resize

def convert_opset_10_to_11(model: onnx.ModelProto) -> onnx.ModelProto:
    """Update opset from 10 to 11.

    Args:
        model (onnx.ModelProto): input onnx model.

    Returns:
        onnx.ModelProto: updated onnx model.
    """
    # Check opset
    helper.logger.info("Checking and changing model meta data.")
    if len(model.opset_import) == 0:
        helper.logger.warning("Updating a model with no opset. Please make sure the model is using opset 10.")
        opset = onnx.helper.make_opsetid('', 11)
        model.opset.append(opset)
    else:
        for opset in model.opset_import:
            if len(opset.domain) != 0:
                continue
            elif opset.version == 10:
                opset.version = 11
                break
            elif opset.version >= 11:
                helper.logger.error(f"Converting opset {opset.version} to opset 11. No need to convert.")
                return None
            else:
                helper.logger.error(f"Converting opset {opset.version} to opset 11. Only opset 10 is supported.")
                return None

    helper.logger.info("Adding required information for the model.")
    graph = model.graph
    # deal with empty node name issue
    other.add_name_to_node(graph)
    # simplify the node param type from initializer to constant
    replacing.replace_initializer_with_Constant(graph)

    # Modify the nodes.
    helper.logger.info("Modifying nodes according to the opset change.")
    replace_min_max_attribute_to_const_node_in_clip_node(graph)
    replace_all_attribute_to_const_node_in_pad_node(graph)
    upsampling_to_resize(graph)
    other.topological_sort(graph)

    while len(graph.value_info) > 0:
        graph.value_info.pop()
    model = onnx.utils.polish_model(model)
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage:{} file_in file_out".format(sys.argv[0]))
        exit(1)

    model = onnx.load(sys.argv[1])
    model = convert_opset_10_to_11(model)

    onnx.save(model, sys.argv[2])