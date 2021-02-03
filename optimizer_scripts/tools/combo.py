"""Combo functions that are usually called together.
"""

import logging
import onnx.utils
from onnx import optimizer

from . import helper
from . import other
from . import replacing
from . import eliminating
from . import fusing
from . import constant_folding
from . import removing_transpose
from . import modhelper
from .torch_pattern import torch_pattern_match

def preprocess(model_proto, disable_fuse_bn=False):
    """The most common used functions before other processing.

    :param model_proto: the original model input\\
    :return: the new model after preprocessing

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
    helper.setup_current_opset_version(model_proto)
    eliminating.eliminate_empty_value_infos(model_proto.graph)
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
    other.add_name_to_node(g)
    replacing.replace_initializer_with_Constant(g)
    other.duplicate_param_shared_constant(g)
    other.topological_sort(g)
    m = onnx.utils.polish_model(m)
    g = m.graph
    eliminating.eliminate_Identify_and_Dropout(g)
    eliminating.eliminate_trivial_maxpool(g)
    eliminating.eliminate_no_children_input(g)
    other.format_value_info_shape(g)
    other.topological_sort(g)
    m = other.inference_shapes(m)
    g = m.graph
    if helper.get_current_opset_version() < 10:
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
    m = onnx.utils.polish_model(m)
    g = m.graph
    other.transpose_B_in_Gemm(g)
    fusing.fuse_BN_into_Gemm(g)
    fusing.fuse_BN_with_Reshape_into_Gemm(g)
    fusing.fuse_Gemm_into_Gemm(g)
    fusing.fuse_consecutive_reducemean(g)
    other.duplicate_shared_Flatten(g)
    replacing.replace_average_pool_with_GAP(g)

    m = onnx.utils.polish_model(m)
    g = m.graph

    replacing.replace_Squeeze_with_Reshape(g)
    replacing.replace_Unsqueeze_with_Reshape(g)
    replacing.replace_Reshape_with_Flatten(g)
    replacing.replace_ReduceMean_with_GlobalAveragePool(g)
    replacing.replace_Sum_with_Adds(g)
    other.topological_sort(g)
    return m


def pytorch_constant_folding(m):
    """Constant folding needed by Pytorch exported models. It should be done
    before using onnx optimizers since the dynamic shape structure may affect
    the optimizations.

    :param m: the original model input\\
    :return: the new model after preprocessing
    """
    logging.info("Working on Pytorch constant folding.")
    replacing.replace_shape_with_constant(m.graph)

    # constant_folding
    m = modhelper.inference_shapes(m)
    while constant_folding.constant_folding(m.graph):
        logging.debug("After constant folding jobs.")
        other.topological_sort(m.graph)
        while len(m.graph.value_info) != 0:
            m.graph.value_info.pop()
        m = modhelper.inference_shapes(m)
        replacing.replace_shape_with_constant(m.graph)

    other.topological_sort(m.graph)
    m = torch_pattern_match(m)
    m = optimizer.optimize(m, ['eliminate_deadend'])
    return m


def tensorflow_optimization(m):
    """Optimizations for tf models can be used in most cases.

    :param m: the original model input\\
    :return: the new model after preprocessing

    It includes:

    - eliminate consecutive Cast
    - eliminate cast after input
    - eliminate shape change after input
    - eliminate Reshape cast
    - eliminate Squeeze before Reshape
    - fuse Transpose into Constant
    - replace Shape with Constant
    """
    g = m.graph
    eliminating.eliminate_consecutive_Cast(g)
    fusing.fuse_Transpose_into_Constant(g)
    fusing.fuse_Add_into_Conv(g)
    fusing.fuse_MatMul_and_Add_into_Gemm(g)
    eliminating.eliminate_Cast_after_input(g)
    other.topological_sort(g)

    m = onnx.utils.polish_model(m)
    g = m.graph

    # constant folding
    replacing.replace_shape_with_constant(m.graph)

    while constant_folding.constant_folding(m.graph):
        m = onnx.utils.polish_model(m)
        replacing.replace_shape_with_constant(m.graph)

    eliminating.eliminate_consecutive_reshape(g)
    eliminating.eliminate_Squeeze_before_Reshape(g)
    other.topological_sort(g)
    return m


def postprocess(m):
    """Inference the shape and prepare for export.

    :param m: the original model input\\
    :return: the new model after preprocessing
    """
    m = onnx.utils.polish_model(m)
    eliminating.eliminate_single_input_Concat(m.graph)
    eliminating.eliminate_nop_Maxpool_and_AveragePool(m.graph)

    m = onnx.utils.polish_model(m)

    replacing.replace_depthwise_1x1_with_bn(m.graph)
    m = onnx.utils.polish_model(m)

    # removing transpose
    m = removing_transpose.eliminate_transposes(m)
    m = onnx.utils.polish_model(m)
    removing_transpose.remove_trivial_transpose(m.graph)
    removing_transpose.fuse_Transpose_into_Gemm_weight(m.graph)

    # fuse some nodes
    fusing.fuse_mul_and_add_into_bn(m.graph)
    m = onnx.utils.polish_model(m)
    fusing.fuse_mul_and_add_into_gemm(m.graph)
    m = onnx.utils.polish_model(m)
    fusing.fuse_conv_and_add_into_conv(m.graph)
    replacing.replace_mul_to_bn(m.graph)

    other.add_output_to_value_info(m.graph)
    m.producer_name = 'kneron_formatter'
    return m
