from tools import other, helper
from onnx import optimizer
import sys
import onnx
import onnx.helper

# StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/time_distributed_564/Reshape_1_kn
def change_reshape_size(g):
    reshape_node = helper.find_node_by_node_name(g, "StatefulPartitionedCall/nn_cut_lstm_0607_addmini_8e5_alldata_25f_25filt_4c_at55dB_cnn_45r_oneoutput2/time_distributed_564/Reshape_1_kn")
    reshape_output = helper.find_value_by_name(g, reshape_node.output[0])
    flatten_node = onnx.helper.make_node(
        "Flatten",
        [reshape_node.input[0]],
        [reshape_node.output[0]],
        name=reshape_node.name,
        axis=1
        )
    g.node.remove(reshape_node)
    g.node.append(flatten_node)
    g.value_info.remove(reshape_output)


m = onnx.load(sys.argv[1])
change_reshape_size(m.graph)
other.topological_sort(m.graph)
m = other.inference_shapes(m)
m = optimizer.optimize(m, ['eliminate_deadend'])
onnx.save(m, sys.argv[2])