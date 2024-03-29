"""Combo functions that are usually called together.
"""

import logging
import onnx.utils
from onnx import optimizer

from . import checker
from . import helper
from . import modhelper
from . import defusing
from . import other
from . import replacing
from . import eliminating
from . import fusing
from . import constant_folding
from . import removing_transpose
from . import expand_lstm
from .common_pattern import torch_pattern_match, tf_pattern_match
from .helper import logger

def preprocess(model_proto, disable_fuse_bn=False, duplicate_shared_weights=True):
    """The most common used functions before other processing.

    Args:
        model_proto: the original model input
        duplicate_shared_weights(bool, optional): duplicate shared weights. Defaults to True.

    Return:
        the new model after preprocessing

    It includes:

    - inference shapes
    - optimize model by ONNX library
    - give names to the nodes
    - replace initializer with Constant node
    - replace -1 batch size with 1
    - eliminate dropout and identity
    - eliminate no children inputs
    - topological sort

    The optimizations provided by ONNX:

    - eliminate_identity
    - eliminate_nop_dropout
    - eliminate_nop_transpose
    - eliminate_nop_pad
    - eliminate_unused_initializer
    - eliminate_deadend
    - fuse_consecutive_squeezes
    - fuse_consecutive_transposes
    - fuse_add_bias_into_conv
    - fuse_transpose_into_gemm
    - fuse_matmul_add_bias_into_gemm
    - fuse_bn_into_conv
    - fuse_pad_into_conv

    """
    logger.info("Preprocessing the model...")
    checker.check_operator_type(model_proto.graph)
    modhelper.setup_current_opset_version(model_proto)
    eliminating.eliminate_empty_value_infos(model_proto.graph)
    other.add_name_to_node(model_proto.graph)
    other.rename_all_node_name(model_proto.graph)
    other.convert_opset12_constants(model_proto.graph)
    defusing.defuse_Einsum(model_proto.graph)
    defusing.defuse_ReduceSum(model_proto.graph)
    defusing.defuse_Conv3D(model_proto.graph)
    other.change_input_output_float16_to_float32(model_proto.graph)
    replacing.replace_initializer_with_Constant(model_proto.graph)
    other.topological_sort(model_proto.graph)
    model_proto = onnx.utils.polish_model(model_proto)
    fusing.fuse_Mul_ReduceSum_into_MatMul(model_proto.graph)
    m = onnx.utils.polish_model(model_proto)
    passes = ['extract_constant_to_initializer',
              'eliminate_nop_dropout',
              'eliminate_deadend',
              'fuse_matmul_add_bias_into_gemm',
              'fuse_pad_into_conv']
    if not disable_fuse_bn:
        passes.append('fuse_bn_into_conv')
    m = optimizer.optimize(m, passes)
    g = m.graph
    # Add name again since onnx optimizer higher than 1.7 may remove node names.
    other.add_name_to_node(g)
    if duplicate_shared_weights:
        replacing.replace_initializer_with_Constant(g, duplicate_shared_weights=True)
        other.duplicate_param_shared_constant(g)
    else:
        replacing.replace_initializer_with_Constant(g, duplicate_shared_weights=False)
    other.topological_sort(g)
    m = onnx.utils.polish_model(m)
    m = expand_lstm.expand_lstm_like_nodes(m)
    g = m.graph
    eliminating.eliminate_consecutive_Cast(m.graph)
    eliminating.eliminate_Cast_after_input(m.graph)
    eliminating.eliminate_nop_pads(g)
    eliminating.eliminate_nop_cast(g)
    eliminating.eliminate_Identify_and_Dropout(g)
    eliminating.eliminate_trivial_maxpool(g)
    eliminating.eliminate_no_children_input(g)
    eliminating.eliminate_Expand_followed_by_broadcast_nodes(g)
    other.format_value_info_shape(g)
    other.topological_sort(g)
    replacing.replace_unsupported_float16_constant_nodes(g)
    m = other.inference_shapes_until_complete_all(m)
    g = m.graph
    replacing.replace_split_with_slices(g)
    other.topological_sort(g)

    return m


def common_optimization(m):
    """Common optimizations can be used in most cases.

    :param m: the original model input\\
    :return: the new model after preprocessing

    It includes:

    - transpose B in Gemm
    - fuse BN into Gemm
    - fuse consecutive Gemm
    - replace AveragePool with GAP
    - replace Squeeze/Unsqueeze with Reshape
    - replace Reshape with Flatten
    """
    logger.info("Doing nodes fusion and replacement... ")
    m = onnx.utils.polish_model(m)
    g = m.graph
    other.transpose_B_in_Gemm(g)
    fusing.fuse_BN_into_Gemm(g)
    fusing.fuse_BN_with_Reshape_into_Gemm(g)
    fusing.fuse_Gemm_into_Gemm(g)
    fusing.fuse_consecutive_reducemean(g)
    fusing.fuse_slice_nodes_into_conv(g)
    fusing.fuse_relu_min_into_clip(g)
    other.duplicate_shared_Flatten(g)
    replacing.replace_average_pool_with_GAP(g)

    m = onnx.utils.polish_model(m)
    m = other.inference_shapes(m)
    g = m.graph

    replacing.replace_Squeeze_with_Reshape(g)
    replacing.replace_Unsqueeze_with_Reshape(g)
    replacing.replace_ReduceMean_with_GlobalAveragePool(g)
    replacing.replace_Sum_with_Adds(g)
    replacing.replace_constant_input_concat_with_pad(g)
    replacing.replace_Gather_with_Reshape(g)
    replacing.replace_Expand_with_Reshape(g)
    fusing.fuse_consecutive_transposes(g)
    other.topological_sort(g)
    return m


def pytorch_constant_folding(m):
    """Constant folding needed by Pytorch exported models. It should be done
    before using onnx optimizers since the dynamic shape structure may affect
    the optimizations.

    :param m: the original model input\\
    :return: the new model after preprocessing
    """
    logger.info("Working on constant folding.")
    m = other.inference_shapes(m)
    replacing.replace_shape_with_constant(m.graph)
    replacing.replace_ConstantOfShape_with_constant(m.graph)

    # constant_folding
    m = other.inference_shapes(m)
    while constant_folding.constant_folding(m.graph):
        logging.debug("After constant folding jobs.")
        other.topological_sort(m.graph)
        while len(m.graph.value_info) != 0:
            m.graph.value_info.pop()

        m = other.inference_shapes(m)
        replacing.replace_shape_with_constant(m.graph)
        replacing.replace_ConstantOfShape_with_constant(m.graph)
    other.topological_sort(m.graph)
    m = other.inference_shapes(m)
    m = torch_pattern_match(m)
    replacing.replace_Gather_with_Slice(m.graph)
    while len(m.graph.value_info) != 0:
        m.graph.value_info.pop()
    m = other.inference_shapes(m)
    m = optimizer.optimize(m, ['eliminate_deadend'])
    return m


def tensorflow_optimization(m):
    """Optimizations for tf models can be used in most cases.

    :param m: the original model input\\
    :return: the new model after preprocessing

    It includes:

    - eliminate shape change after input
    - eliminate Reshape cast
    - eliminate Squeeze before Reshape
    - fuse Transpose into Constant
    - replace Shape with Constant
    """

    fusing.fuse_Transpose_into_Constant(m.graph)
    fusing.fuse_MatMul_and_Add_into_Gemm(m.graph)
    other.topological_sort(m.graph)

    m = onnx.utils.polish_model(m)

    # constant folding
    replacing.replace_shape_with_constant(m.graph)

    # constant_folding
    m = other.inference_shapes(m)
    while constant_folding.constant_folding(m.graph):
        logging.debug("After constant folding jobs.")
        other.topological_sort(m.graph)
        while len(m.graph.value_info) != 0:
            m.graph.value_info.pop()

        m = other.inference_shapes(m)
        replacing.replace_shape_with_constant(m.graph)
    other.topological_sort(m.graph)
    m = tf_pattern_match(m)
    m = optimizer.optimize(m, ['eliminate_deadend'])

    while len(m.graph.value_info) != 0:
        m.graph.value_info.pop()
    m = other.inference_shapes(m)
    m = optimizer.optimize(m, ['eliminate_deadend'])
    return m


def postprocess(m):
    """Inference the shape and prepare for export.

    :param m: the original model input\\
    :return: the new model after preprocessing
    """
    logger.info("Postprocessing the model...")
    while len(m.graph.value_info) > 0:
        m.graph.value_info.pop()
    m = onnx.utils.polish_model(m)
    eliminating.eliminate_single_input_Concat(m.graph)
    eliminating.eliminate_nop_Maxpool_and_AveragePool(m.graph)
    eliminating.eliminate_trivial_elementwise_calculation(m.graph)
    m = onnx.utils.polish_model(m)

    replacing.replace_depthwise_1x1_with_bn(m.graph)
    m = onnx.utils.polish_model(m)

    # removing transpose
    m = removing_transpose.eliminate_transposes(m)
    m = onnx.utils.polish_model(m)
    removing_transpose.remove_trivial_transpose(m.graph)
    removing_transpose.fuse_Transpose_into_Gemm_weight(m.graph)

    # removing reshapes
    other.move_4D_to_5D_Reshape(m.graph)
    eliminating.eliminate_consecutive_reshape_like_nodes(m.graph)
    eliminating.eliminate_nop_reshape(m.graph)
    eliminating.eliminate_nop_flatten(m.graph)
    replacing.replace_Reshape_with_Flatten(m.graph)
    other.topological_sort(m.graph)

    # fuse some nodes
    defusing.defuse_div_with_reciprocal_and_mul(m.graph)
    fusing.fuse_mul_and_add_into_bn(m.graph)
    m = onnx.utils.polish_model(m)
    fusing.fuse_mul_and_add_into_gemm(m.graph)
    m = onnx.utils.polish_model(m)
    fusing.fuse_conv_and_add_into_conv(m.graph)
    m = onnx.utils.polish_model(m)
    replacing.replace_mul_to_bn(m.graph)
    replacing.replace_div_to_bn(m.graph)
    replacing.replace_add_to_bn(m.graph)
    replacing.replace_sub_to_bn(m.graph)
    replacing.replace_sub_with_bn_and_add(m.graph)
    m = onnx.utils.polish_model(m)
    m = other.inference_shapes_until_complete_all(m)

    other.add_output_to_value_info(m.graph)
    m = optimizer.optimize(m, ['eliminate_deadend'])
    m.producer_name = 'kneron_formatter'
    return m
