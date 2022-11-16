import onnx
import onnx.helper
import numpy as np

from tools import helper
from tools import other
from tools import modhelper


def replace_input(g):
    # Remove node and input
    gather_node = helper.find_node_by_node_name(g, "StatefulPartitionedCall/sequential/embedding/embedding_lookup")
    cast_node = helper.find_node_by_node_name(g, "StatefulPartitionedCall/sequential/embedding/Cast")
    g.node.remove(gather_node)
    g.node.remove(cast_node)
    g.input.pop()
    # Add new input
    new_input = onnx.helper.make_tensor_value_info(
        "StatefulPartitionedCall/sequential/embedding/embedding_lookup:0",
        1,
        (1, 50, 16)
    )
    g.input.append(new_input)


def replace_shape_at_beginning(g):
    a = np.zeros(shape=(1, 16), dtype=np.float32)
    new_constant = helper.numpy_to_constant('StatefulPartitionedCall/sequential/lstm/zeros:0', a)
    names_to_remove = [
        'StatefulPartitionedCall/sequential/lstm/Shape',
        'StatefulPartitionedCall/sequential/lstm/Shape__37',
        'StatefulPartitionedCall/sequential/lstm/strided_slice',
        'StatefulPartitionedCall/sequential/lstm/zeros/packed_Concat__48',
        'StatefulPartitionedCall/sequential/lstm/zeros_1__45',
        'StatefulPartitionedCall/sequential/lstm/zeros'
    ]
    nodes_to_remove = [helper.find_node_by_node_name(g, name) for name in names_to_remove]
    for node in nodes_to_remove:
        g.node.remove(node)
    g.node.insert(0, new_constant)


def delete_reshapes_before_concat(g):
    concat_node = helper.find_node_by_node_name(g, "StatefulPartitionedCall/sequential/lstm/while_loop:7")
    input_reshape_nodes = [helper.find_node_by_output_name(g, name) for name in concat_node.input]
    value_info_to_remove = [helper.find_value_by_name(g, name) for name in concat_node.input]
    constant_nodes = [helper.find_node_by_output_name(g, node.input[1]) for node in input_reshape_nodes]
    # Replace concat inputs
    for reshape_node in input_reshape_nodes:
        for node in helper.find_following_nodes_by_input_value_name(g, reshape_node.output[0]):
            modhelper.replace_node_input(node, reshape_node.output[0], reshape_node.input[0])
    # Also put concat output value info into remove list
    value_info_to_remove.append(helper.find_value_by_name(g, concat_node.output[0]))
    for node in constant_nodes:
        value_info_to_remove.append(helper.find_value_by_name(g, node.output[0]))
    # Remove nodes
    for node in constant_nodes:
        if len(helper.find_following_nodes_by_input_value_name(g, node.output[0])) == 1:
            g.node.remove(node)
    for node in input_reshape_nodes:
        g.node.remove(node)
    for value in value_info_to_remove:
        if value is None:
            continue
        else:
            g.value_info.remove(value)


if __name__ == '__main__':
    # src_onnx_path = "/home/kneron/Downloads/hate_speech_mod/hate_speech.onnx"
    # src_onnx_m = onnx.load(src_onnx_path)
    # # After Gather, the shape should be 1x50x16
    # g = src_onnx_m.graph
    # replace_input(g)
    # replace_shape_at_beginning(g)
    # other.change_output_shape(g, ['dense_2 1 1'])
    # onnx.save(src_onnx_m, "/home/kneron/Downloads/hate_speech_mod/hate_speech.1.onnx")

    # Reset shapes
    # src_onnx_path = "/home/kneron/Downloads/hate_speech_mod/hate_speech.2.onnx"
    # src_onnx_m = onnx.load(src_onnx_path)
    # # After Gather, the shape should be 1x50x16
    # g = src_onnx_m.graph
    # reshape_node = helper.find_node_by_node_name(g, 'StatefulPartitionedCall/sequential/flatten/Reshape')
    # modhelper.replace_node_input(reshape_node, 'StatefulPartitionedCall/sequential/lstm/transpose_1:0', 'StatefulPartitionedCall/sequential/lstm/while_loop:7')
    # transpose_node = helper.find_node_by_node_name(g, 'StatefulPartitionedCall/sequential/lstm/transpose_1')
    # g.node.remove(transpose_node)
    # onnx.save(src_onnx_m, "/home/kneron/Downloads/hate_speech_mod/hate_speech.3.onnx")

    # Run outside the script: python pytorch_exported_onnx_preprocess.py ~/Downloads/hate_speech_mod/hate_speech.3.onnx ~/Downloads/hate_speech_mod/hate_speech.5.onnx
    om = onnx.load("hate_speech.5.onnx")
    om.opset_import.pop()
    onnx.save(om, "hate_speech.6.onnx")

    # Remove reshapes
    src_onnx_path = "/home/kneron/Downloads/hate_speech_mod/hate_speech.6.onnx"
    src_onnx_m = onnx.load(src_onnx_path)
    delete_reshapes_before_concat(src_onnx_m.graph)
    m = other.inference_shapes(src_onnx_m)
    # m.opset_import.pop()
    onnx.save(m, "/home/kneron/Downloads/hate_speech_mod/hate_speech.7.onnx")

