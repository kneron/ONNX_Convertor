import onnx.helper
import numpy as np
from . import helper
from .other import topological_sort
from .modhelper import delete_value_with_name_if_exists

def fuse_Transpose_into_Constant(g):
    """
    Fuse Transpose layers into the Constant layers before

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Transpose':
            continue
        prev_node = helper.find_node_by_output_name(g, node.input[0])
        if prev_node is None or prev_node.op_type != 'Constant':
            continue

        pre_shape, data_list = helper.constant_to_list(prev_node)
        w = np.reshape(data_list, pre_shape)
        w = w.transpose(node.attribute[0].ints)
        new_shape = w.shape
        w = w.flatten()

        new_tensor = onnx.helper.make_tensor(
            name=prev_node.name+'_data',
            data_type=prev_node.attribute[0].t.data_type,
            dims=new_shape,
            vals=w.tolist()
        )
        new_node = onnx.helper.make_node(
            'Constant',
            [],
            [node.output[0]],
            name=node.output[0],
            value=new_tensor
        )

        value_between = helper.find_value_by_name(g, prev_node.output[0])
        value_type = value_between.type.tensor_type.elem_type
        g.value_info.remove(value_between)

        g.node.extend([new_node])
        node_to_remove.append(node)
        node_to_remove.append(prev_node)

        if new_node.output[0] not in [i.name for i in g.value_info]:
            new_value = onnx.helper.make_tensor_value_info(
                name=new_node.output[0],
                elem_type=value_type,
                shape=new_shape
                )
            g.value_info.extend([new_value])
            if new_node.output[0]:
                val_info_to_del = helper.find_value_by_name(g, new_node.output[0])
                g.value_info.remove(val_info_to_del)

    for node in node_to_remove:
        g.node.remove(node)

    topological_sort(g)

def fuse_Add_into_Conv(g):
    """
    Fuse Transpose layers into the Constant layers before

    :param g: the onnx graph
    """
    node_to_remove = []
    for node in g.node:
        if node.op_type != 'Add':
            continue
        conv_node = helper.find_node_by_output_name(g, node.input[0])
        cons_node = helper.find_node_by_output_name(g, node.input[1])
        if conv_node is None or cons_node is None:
            continue
        if conv_node.op_type != 'Conv' or cons_node.op_type != 'Constant':
            continue
        if len(conv_node.input) > 2:
            continue
        # This layer should be fused. Connect constant node into convolution node.
        add_node = node
        conv_node.input.extend([cons_node.output[0]])
        old_value = helper.find_value_by_name(g, conv_node.output[0])
        conv_node.output[0] = add_node.output[0]
        # Remove origin conv_node_output
        g.value_info.remove(old_value)
        # Remove current node
        node_to_remove.append(add_node)
    # Apply changes to the model
    for node in node_to_remove:
        g.node.remove(node)

def fuse_BN_into_Gemm(g):
    """Fuse the following BN into the previous Gemm.

    :param g: the graph
    """
    node_to_remove = []
    for node in g.node:
        # Check for BN and Gemm
        if node.op_type != 'BatchNormalization':
            continue
        gemm_node = helper.find_node_by_output_name(g, node.input[0])
        if gemm_node is None:
            continue
        if gemm_node.op_type != 'Gemm':
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, gemm_node.output[0])) > 1:
            continue
        bn_node = node
        # Get original weights
        gemm_b_node = helper.find_node_by_output_name(g, gemm_node.input[1])
        gemm_b = helper.constant_to_numpy(gemm_b_node)
        gemm_c_node = helper.find_node_by_output_name(g, gemm_node.input[2])
        gemm_c = helper.constant_to_numpy(gemm_c_node)
        bn_scale_node = helper.find_node_by_output_name(g, bn_node.input[1])
        bn_scale = helper.constant_to_numpy(bn_scale_node)
        bn_bias_node = helper.find_node_by_output_name(g, bn_node.input[2])
        bn_bias = helper.constant_to_numpy(bn_bias_node)
        bn_mean_node = helper.find_node_by_output_name(g, bn_node.input[3])
        bn_mean = helper.constant_to_numpy(bn_mean_node)
        bn_var_node = helper.find_node_by_output_name(g, bn_node.input[4])
        bn_var = helper.constant_to_numpy(bn_var_node)
        # Apply attributes
        # epsilon
        epsilon = helper.get_attribute_by_name(bn_node, 'epsilon')
        if epsilon is None:
            epsilon = 0.00001
        else:
            epsilon = epsilon.f
        bn_var = bn_var + epsilon
        # alpha
        alpha = helper.get_attribute_by_name(gemm_node, 'alpha')
        if alpha is None:
            alpha = 1
        else:
            alpha = alpha.f
        gemm_b = gemm_b * alpha
        # beta
        beta = helper.get_attribute_by_name(gemm_node, 'beta')
        if beta is None:
            beta = 1
        else:
            beta = beta.f
        gemm_c = gemm_c * beta
        # transA
        transA = helper.get_attribute_by_name(gemm_node, 'transA')
        if transA is not None and transA.i == 1:
            raise RuntimeError("Do not support transA")
        # transB
        transB = helper.get_attribute_by_name(gemm_node, 'transB')
        if transB is not None and transB.i == 1:
            gemm_b = gemm_b.transpose()
        # Calculate new weights
        new_gemm_b = gemm_b * bn_scale / np.sqrt(bn_var)
        new_gemm_c = (gemm_c - bn_mean) * bn_scale / np.sqrt(bn_var) + bn_bias
        # Replace original weights
        new_gemm_b_node = helper.numpy_to_constant(gemm_b_node.name + '_fused', new_gemm_b)
        new_gemm_c_node = helper.numpy_to_constant(gemm_c_node.name + '_fused', new_gemm_c)
        g.node.extend([new_gemm_b_node, new_gemm_c_node])
        node_to_remove.extend([gemm_b_node,
                               gemm_c_node,
                               bn_node,
                               bn_scale_node,
                               bn_bias_node,
                               bn_mean_node,
                               bn_var_node])
        # Modify attributes
        # alpha
        alpha = helper.get_attribute_by_name(gemm_node, 'alpha')
        if alpha is not None:
            alpha.f = 1.0
        # beta
        beta = helper.get_attribute_by_name(gemm_node, 'beta')
        if beta is not None:
            beta.f = 1.0
        # transB
        transB = helper.get_attribute_by_name(gemm_node, 'transB')
        if transB is not None:
            transB.i = 0
        # Connect the new graph
        gemm_node.input[1] = new_gemm_b_node.output[0]
        gemm_node.input[2] = new_gemm_c_node.output[0]
        gemm_b_value = helper.find_value_by_name(g, gemm_b_node.output[0])
        gemm_c_value = helper.find_value_by_name(g, gemm_c_node.output[0])
        gemm_b_value.name = new_gemm_b_node.output[0]
        gemm_c_value.name = new_gemm_c_node.output[0]
        gemm_value = helper.find_value_by_name(g, gemm_node.output[0])
        g.value_info.remove(gemm_value)
        gemm_node.output[0] = bn_node.output[0]
        for i in range(1, 5):
            value = helper.find_value_by_name(g, bn_node.input[i])
            g.value_info.remove(value)
    # Remove useless nodes
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)

def fuse_BN_with_Reshape_into_Gemm(g):
    """Fuse the following BN into the previous Gemm, even with Reshape or \\
        Squeeze and Unsqueeze surrounding.

    :param g: the graph
    """
    node_to_remove = []
    for node in g.node:
        # Check for BN and Gemm pattern: Gemm A BN B
        # Find BatchNorm Node
        if node.op_type != 'BatchNormalization':
            continue
        bn_node = node
        # Find A Node
        a_node = helper.find_node_by_output_name(g, node.input[0])
        if a_node is None or len(a_node.input) == 0:
            continue
        # Find Gemm Node
        gemm_node = helper.find_node_by_output_name(g, a_node.input[0])
        if gemm_node is None or gemm_node.op_type != 'Gemm':
            continue
        # Find B Node
        b_node_list = helper.find_following_nodes_by_input_value_name(g, bn_node.output[0])
        if len(b_node_list) == 0:
            the_output = helper.find_output_by_name(g, bn_node.output[0])
            if the_output is None:
                continue
            b_node = None
        elif len(b_node_list) > 1:
            continue
        else:
            b_node = b_node_list[0]
        # Check for branches
        if len(helper.find_following_nodes_by_input_value_name(g, gemm_node.output[0])) > 1:
            continue
        if len(helper.find_following_nodes_by_input_value_name(g, a_node.output[0])) > 1:
            continue
        # Check type of A
        if a_node.op_type == 'Unsqueeze':
            axes = helper.get_attribute_by_name(a_node, 'axes')
            if axes.ints != [2]:
                continue
        elif a_node.op_type == 'Reshape':
            a = helper.constant_to_list(helper.find_node_by_output_name(g, a_node.input[1]))[1]
            if len(a) != 3 or a[2] != 1:
                continue
        else:
            continue
        # Check type of B
        if b_node is None:
            pass
        elif b_node.op_type == 'Flatten':
            pass
        elif b_node.op_type == 'Squeeze':
            axes = helper.get_attribute_by_name(a_node, 'axes')
            if axes.ints != [2]:
                continue
        elif b_node.op_type == 'Reshape':
            a = helper.constant_to_list(helper.find_node_by_output_name(g, b_node.input[1]))[1]
            if len(a) != 2:
                continue
        else:
            continue
        # Construct new Nodes
        # Get original weights
        gemm_b_node = helper.find_node_by_output_name(g, gemm_node.input[1])
        gemm_b = helper.constant_to_numpy(gemm_b_node)
        gemm_c_node = helper.find_node_by_output_name(g, gemm_node.input[2])
        gemm_c = helper.constant_to_numpy(gemm_c_node)
        bn_scale_node = helper.find_node_by_output_name(g, bn_node.input[1])
        bn_scale = helper.constant_to_numpy(bn_scale_node)
        bn_bias_node = helper.find_node_by_output_name(g, bn_node.input[2])
        bn_bias = helper.constant_to_numpy(bn_bias_node)
        bn_mean_node = helper.find_node_by_output_name(g, bn_node.input[3])
        bn_mean = helper.constant_to_numpy(bn_mean_node)
        bn_var_node = helper.find_node_by_output_name(g, bn_node.input[4])
        bn_var = helper.constant_to_numpy(bn_var_node)
        # Apply attributes
        # epsilon
        epsilon = helper.get_attribute_by_name(bn_node, 'epsilon')
        if epsilon is None:
            epsilon = 0.00001
        else:
            epsilon = epsilon.f
        bn_var = bn_var + epsilon
        # alpha
        alpha = helper.get_attribute_by_name(gemm_node, 'alpha')
        if alpha is None:
            alpha = 1
        else:
            alpha = alpha.f
        gemm_b = gemm_b * alpha
        # beta
        beta = helper.get_attribute_by_name(gemm_node, 'beta')
        if beta is None:
            beta = 1
        else:
            beta = beta.f
        gemm_c = gemm_c * beta
        # transA
        transA = helper.get_attribute_by_name(gemm_node, 'transA')
        if transA is not None and transA.i == 1:
            raise RuntimeError("Do not support transA")
        # transB
        transB = helper.get_attribute_by_name(gemm_node, 'transB')
        if transB is not None and transB.i == 1:
            gemm_b = gemm_b.transpose()
        # Calculate new weights
        new_gemm_b = gemm_b * bn_scale / np.sqrt(bn_var)
        new_gemm_c = (gemm_c - bn_mean) * bn_scale / np.sqrt(bn_var) + bn_bias
        # Replace original weights
        new_gemm_b_node = helper.numpy_to_constant(gemm_b_node.name + '_fused', new_gemm_b)
        new_gemm_c_node = helper.numpy_to_constant(gemm_c_node.name + '_fused', new_gemm_c)
        g.node.extend([new_gemm_b_node, new_gemm_c_node])
        # Modify attributes
        # alpha
        alpha = helper.get_attribute_by_name(gemm_node, 'alpha')
        if alpha is not None:
            alpha.f = 1.0
        # beta
        beta = helper.get_attribute_by_name(gemm_node, 'beta')
        if beta is not None:
            beta.f = 1.0
        # transB
        transB = helper.get_attribute_by_name(gemm_node, 'transB')
        if transB is not None:
            transB.i = 0
        # Remove useless nodes
        node_to_remove.extend([gemm_b_node,
                               gemm_c_node,
                               bn_node,
                               bn_scale_node,
                               bn_bias_node,
                               bn_mean_node,
                               bn_var_node,
                               a_node])
        if a_node.op_type == 'Reshape':
            node_to_remove.append(helper.find_node_by_output_name(g, a_node.input[1]))
        if b_node is not None:
            node_to_remove.append(b_node)
            if b_node.op_type == 'Reshape':
                node_to_remove.append(helper.find_node_by_output_name(g, b_node.input[1]))
        # Delete useless value infos
        value = helper.find_value_by_name(g, a_node.output[0])
        g.value_info.remove(value)
        if a_node.op_type == 'Reshape':
            value = helper.find_value_by_name(g, a_node.input[1])
            g.value_info.remove(value)
        for i in range(1, 5):
            value = helper.find_value_by_name(g, bn_node.input[i])
            g.value_info.remove(value)
        value = helper.find_value_by_name(g, bn_node.output[0])
        if value is not None:
            g.value_info.remove(value)
        if b_node is not None:
            value = helper.find_value_by_name(g, gemm_node.output[0])
            g.value_info.remove(value)
            if b_node.op_type == 'Reshape':
                value = helper.find_value_by_name(g, b_node.input[1])
                g.value_info.remove(value)
        # Connect the new graph
        # Connect Gemm new weights
        gemm_node.input[1] = new_gemm_b_node.output[0]
        gemm_node.input[2] = new_gemm_c_node.output[0]
        gemm_b_value = helper.find_value_by_name(g, gemm_b_node.output[0])
        gemm_c_value = helper.find_value_by_name(g, gemm_c_node.output[0])
        gemm_b_value.name = new_gemm_b_node.output[0]
        gemm_b_value.type.tensor_type.shape.dim[0].dim_value = new_gemm_b.shape[0]
        gemm_b_value.type.tensor_type.shape.dim[1].dim_value = new_gemm_b.shape[1]
        gemm_c_value.name = new_gemm_c_node.output[0]
        if b_node is None:
            # If b node is None, set the Gemm output as the graph output
            output_value = helper.find_output_by_name(g, bn_node.output[0])
            g.output.remove(output_value)
            g.output.extend([helper.find_value_by_name(g, gemm_node.output[0])])
        else:
            # Else, set node B output as gemm output
            gemm_node.output[0] = b_node.output[0]
    # Remove useless nodes
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)


def fuse_Gemm_into_Gemm(g):
    """Fuse the previous Gemm into the following Gemm.

    :param g: the graph
    """
    node_to_remove = []
    for node in g.node:
        # Check for Gemm and Gemm
        if node.op_type != 'Gemm':
            continue
        prev_node = helper.find_node_by_output_name(g, node.input[0])
        if prev_node is None:
            continue
        if prev_node.op_type != 'Gemm':
            continue
        # Get original weights
        prev_b_node = helper.find_node_by_output_name(g, prev_node.input[1])
        prev_b = helper.constant_to_numpy(prev_b_node)
        prev_c_node = helper.find_node_by_output_name(g, prev_node.input[2])
        prev_c = helper.constant_to_numpy(prev_c_node)
        b_node = helper.find_node_by_output_name(g, node.input[1])
        b = helper.constant_to_numpy(b_node)
        c_node = helper.find_node_by_output_name(g, node.input[2])
        c = helper.constant_to_numpy(c_node)
        # Apply attributes
        # alpha
        alpha = helper.get_attribute_by_name(node, 'alpha')
        if alpha is None:
            alpha = 1
        else:
            alpha = alpha.f
        b = b * alpha
        alpha = helper.get_attribute_by_name(prev_node, 'alpha')
        if alpha is None:
            alpha = 1
        else:
            alpha = alpha.f
        prev_b = prev_b * alpha
        # beta
        beta = helper.get_attribute_by_name(node, 'beta')
        if beta is None:
            beta = 1
        else:
            beta = beta.f
        c = c * beta
        beta = helper.get_attribute_by_name(prev_node, 'beta')
        if beta is None:
            beta = 1
        else:
            beta = beta.f
        prev_c = prev_c * beta
        # transA
        transA = helper.get_attribute_by_name(node, 'transA')
        if transA is not None and transA.i == 1:
            raise RuntimeError("Do not support transA")
        transA = helper.get_attribute_by_name(prev_node, 'transA')
        if transA is not None and transA.i == 1:
            raise RuntimeError("Do not support transA")
        # transB
        transB = helper.get_attribute_by_name(node, 'transB')
        if transB is not None and transB.i == 1:
            b = b.transpose()
        transB = helper.get_attribute_by_name(prev_node, 'transB')
        if transB is not None and transB.i == 1:
            prev_b = prev_b.transpose()
        # Calculate new weights
        new_b = prev_b.dot(b)
        new_c = prev_c.dot(b) + c
        # Replace original weights
        new_b_node = helper.numpy_to_constant(b_node.name + '_fused', new_b)
        new_c_node = helper.numpy_to_constant(c_node.name + '_fused', new_c)
        g.node.extend([new_b_node, new_c_node])
        node_to_remove.extend([b_node,
                               c_node,
                               prev_b_node,
                               prev_c_node,
                               prev_node])
        # Modify attributes
        # alpha
        alpha = helper.get_attribute_by_name(node, 'alpha')
        if alpha is not None:
            alpha.f = 1.0
        # beta
        beta = helper.get_attribute_by_name(node, 'beta')
        if beta is not None:
            beta.f = 1.0
        # transB
        transB = helper.get_attribute_by_name(node, 'transB')
        if transB is not None:
            transB.i = 0
        # Connect the new graph
        node.input[0] = prev_node.input[0]
        delete_value_with_name_if_exists(g, prev_node.output[0])
        for i in range(1, 3):
            delete_value_with_name_if_exists(g, prev_node.input[i])
            delete_value_with_name_if_exists(g, node.input[i])
        node.input[1] = new_b_node.output[0]
        node.input[2] = new_c_node.output[0]
    # Remove useless nodes
    for node in node_to_remove:
        g.node.remove(node)
    topological_sort(g)

def fuse_MatMul_and_Add_into_Gemm(g):
    """
    Fuse MatMul and Add layers into a new Gemm layers.

    :param g: the onnx graph
    :raises ValueError: MatMul must be followed by an Add node
    """
    node_to_remove = []
    node_to_add = []
    for node in g.node:
        if node.op_type != 'MatMul':
            continue
        add_node = None
        for i in g.node:
            if not i.input:
                continue
            if i.input[0] == node.output[0]:
                add_node = i
                break
        value_to_remove = helper.find_value_by_name(g, node.output[0])
        if add_node is None or value_to_remove is None or add_node.op_type != 'Add':
            continue
        input_list = node.input
        input_list.append(add_node.input[1]),
        new_node = onnx.helper.make_node(
            "Gemm",
            input_list,
            add_node.output,
            name=node.name,
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0
        )
        node_to_add.append(new_node)
        node_to_remove.append(node)
        node_to_remove.append(add_node)
        g.value_info.remove(value_to_remove)
    for node in node_to_remove:
        g.node.remove(node)
    g.node.extend(node_to_add)

def fuse_consecutive_transposes(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Transpose':
            continue
        pre_node = helper.find_node_by_output_name(g, node.input[0])
        if pre_node.op_type != 'Transpose':
            continue

        pre_permutation = list(pre_node.attribute[0].ints)
        cur_permutation = list(node.attribute[0].ints)
        if len(pre_permutation) != len(cur_permutation):
            continue

        new_permutation = []
        for ind in cur_permutation:
            new_permutation.append(pre_permutation[ind])

        new_trans_node = onnx.helper.make_node(
            'Transpose',
            [pre_node.input[0]],
            [node.output[0]],
            name=node.name,
            perm=new_permutation
        )

        g.node.extend([new_trans_node])
        node_to_del.extend([pre_node, node])

        mid_val_info = helper.find_value_by_name(g, node.input[0])
        if mid_val_info:
            g.value_info.remove(mid_val_info)

    while node_to_del:
        node = node_to_del.pop()
        g.node.remove(node)

    topological_sort(g)

def fuse_mul_and_add_into_bn(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Add':
            continue
        add_node = node
        input_nodes_add = [helper.find_node_by_output_name(g, input_name) for input_name in add_node.input]
        if any([n == None for n in input_nodes_add]):
            continue
        mul_node, const_add = None, None
        for input_node_add in input_nodes_add:
            if input_node_add.op_type == 'Mul':
                mul_node = input_node_add
            elif input_node_add.op_type == 'Constant':
                const_add = input_node_add
            else:
                pass
        if not mul_node or not const_add:
            continue
        data_input_name, const_mul = None, None
        for input_name in mul_node.input:
            input_node = helper.find_node_by_output_name(g, input_name)
            if not input_node:
                data_input_name = input_name
            elif input_node.op_type == 'Constant':
                if not const_mul:
                    const_mul = input_node
                else:
                    data_input_name = input_name
            else:
                data_input_name = input_name

        if not const_mul:
            continue

        scale_shape, scale_data = helper.constant_to_list(const_mul)
        bias_shape, __ = helper.constant_to_list(const_add)
        c_dim = len(scale_data)
        if scale_shape != bias_shape:
            continue

        _ , previous_node_output_shape = helper.find_size_shape_from_value(helper.find_value_by_name(g, data_input_name))
        # only allow 4 dim data input due to the hardware limitation
        if len(previous_node_output_shape) != 4:
            continue

        # check if mul's dim and input channel dimension are matched
        if previous_node_output_shape[1] != c_dim:
            continue

        if scale_shape == [1, c_dim, 1, 1]:
            # remove all '1'
            for _ in range(3):
                const_add.attribute[0].t.dims.remove(1)
                const_mul.attribute[0].t.dims.remove(1)
        elif scale_shape == [1, c_dim]:
            # remove all '1'
            const_add.attribute[0].t.dims.remove(1)
            const_mul.attribute[0].t.dims.remove(1)
        elif scale_shape == 1 and c_dim == 1:
            # Single value weight
            const_add.attribute[0].t.dims.append(1)
            const_mul.attribute[0].t.dims.append(1)
        else:
            continue

        bn_name = add_node.output[0]
        const_mean = helper.list_to_constant(bn_name+'_mean', [c_dim], [0.0 for _ in range(c_dim)])
        const_var = helper.list_to_constant(bn_name+'_var', [c_dim], [1.0 for _ in range(c_dim)])

        bn_node = onnx.helper.make_node(
            'BatchNormalization',
            [data_input_name, const_mul.output[0], const_add.output[0],\
                const_mean.output[0], const_var.output[0]],
            [add_node.output[0]],
            name=bn_name,
            epsilon=0.00000001
        )

        mid_val_info = helper.find_value_by_name(g, mul_node.output[0])
        scale_val_info = helper.find_value_by_name(g, const_mul.output[0])
        bais_val_info = helper.find_value_by_name(g, const_add.output[0])
        g.value_info.remove(mid_val_info)
        g.value_info.remove(scale_val_info)
        g.value_info.remove(bais_val_info)

        new_scale_val_info = onnx.helper.make_tensor_value_info(
            const_mul.output[0],
            const_mul.attribute[0].t.data_type,
            [c_dim]
            )
        new_bais_val_info = onnx.helper.make_tensor_value_info(
            const_add.output[0],
            const_add.attribute[0].t.data_type,
            [c_dim]
        )
        mean_val_info = onnx.helper.make_tensor_value_info(
            const_mean.output[0],
            const_mean.attribute[0].t.data_type,
            [c_dim]
        )
        var_val_info = onnx.helper.make_tensor_value_info(
            const_var.output[0],
            const_var.attribute[0].t.data_type,
            [c_dim]
        )

        g.value_info.extend([new_scale_val_info])
        g.value_info.extend([new_bais_val_info])
        g.value_info.extend([mean_val_info])
        g.value_info.extend([var_val_info])
        g.node.extend([bn_node])
        g.node.extend([const_mean])
        g.node.extend([const_var])
        node_to_del.extend([mul_node, add_node])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)


def fuse_mul_and_add_into_gemm(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Add':
            continue
        add_node = node
        mul_node = helper.find_node_by_output_name(g, add_node.input[0])
        if not mul_node or mul_node.op_type != 'Mul':
            continue
        mul_const = helper.find_node_by_output_name(g, mul_node.input[1])
        if not mul_const or mul_const.op_type != 'Constant':
            continue
        add_const = helper.find_node_by_output_name(g, add_node.input[1])
        if not add_const or add_const.op_type != 'Constant':
            continue

        input_val = helper.find_value_by_name(g, mul_node.input[0])
        if not input_val:
            input_val = helper.find_input_by_name(g, mul_node.input[0])
        if not input_val:
            continue

        _, input_shape = helper.find_size_shape_from_value(input_val)
        if not input_shape:
            continue

        dim = int(np.prod(input_shape))
        if input_shape != [1, dim]:
            continue

        mul_const_shape, mul_const_data = helper.constant_to_list(mul_const)
        add_const_shape, __ = helper.constant_to_list(add_const)

        if len(mul_const_shape) != 1 or mul_const_shape[0] != dim:
            continue
        if len(add_const_shape) != 1 or add_const_shape[0] != dim:
            continue

        b_data = np.zeros([dim, dim])
        for i in range(dim):
            b_data[i][i] = mul_const_data[i]
        b_data = b_data.flatten().tolist()
        b_tensor = onnx.helper.make_tensor(
            name=mul_const.name+'_tensor',
            data_type=mul_const.attribute[0].t.data_type,
            dims=[dim, dim],
            vals=b_data
        )
        b_const_node = onnx.helper.make_node(
            'Constant',
            [],
            [mul_const.output[0]],
            value=b_tensor,
            name=mul_const.output[0]
        )

        add_const.attribute[0].t.dims.insert(0, 1)

        gemm_node = onnx.helper.make_node(
            'Gemm',
            [mul_node.input[0], b_const_node.output[0], add_const.output[0]],
            [add_node.output[0]],
            name=add_node.output[0]
        )

        g.node.extend([gemm_node, b_const_node])
        node_to_del.extend([mul_const, mul_node, add_node])

        val_info_mid = helper.find_value_by_name(g, mul_node.output[0])
        val_info_mul_const = helper.find_value_by_name(g, mul_const.output[0])
        val_info_add_const = helper.find_value_by_name(g, add_const.output[0])
        if val_info_mid:
            g.value_info.remove(val_info_mid)
        if val_info_mul_const:
            g.value_info.remove(val_info_mul_const)
        if val_info_add_const:
            g.value_info.remove(val_info_add_const)

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)

def fuse_conv_and_add_into_conv(g):
    node_to_del = []
    for node in g.node:
        if node.op_type != 'Add':
            continue
        add_node = node
        add_const = helper.find_node_by_output_name(g, add_node.input[1])
        if not add_const or add_const.op_type != 'Constant':
            continue

        conv_node = helper.find_node_by_output_name(g, add_node.input[0])
        if not conv_node or conv_node.op_type != 'Conv':
            continue
        weight_node = helper.find_node_by_output_name(g, conv_node.input[1])
        if not weight_node or weight_node.op_type != 'Constant':
            continue

        m_dim = weight_node.attribute[0].t.dims[0]
        if add_const.attribute[0].t.dims != [1, m_dim, 1, 1]:
            continue
        for _ in range(3):
            add_const.attribute[0].t.dims.remove(1)

        conv_node.input.extend([add_const.output[0]])
        conv_node.output.pop()
        conv_node.output.extend([add_node.output[0]])

        node_to_del.append(add_node)

        old_add_const_val_info = helper.find_value_by_name(g, add_node.input[1])
        old_conv_output_val_info = helper.find_value_by_name(g, conv_node.output[0])
        if old_add_const_val_info:
            g.value_info.remove(old_add_const_val_info)
        if old_conv_output_val_info:
            g.value_info.remove(old_conv_output_val_info)

        new_add_const_val_info = onnx.helper.make_tensor_value_info(
            add_const.output[0],
            add_const.attribute[0].t.data_type,
            add_const.attribute[0].t.dims
        )
        g.value_info.extend([new_add_const_val_info])

    while node_to_del:
        g.node.remove(node_to_del.pop())

    topological_sort(g)


def fuse_consecutive_reducemean(g):
    node_to_del = []
    for node in g.node:
        # Find consecutive ReduceMean
        if node.op_type != 'ReduceMean':
            continue
        pre_node = helper.find_node_by_output_name(g, node.input[0])
        if pre_node is None or pre_node.op_type != 'ReduceMean':
            continue
        # Check attributes
        pre_keepdims = helper.get_var_attribute_by_name(pre_node, 'keepdims', 'int')
        pre_axes = helper.get_list_attribute_by_name(pre_node, 'axes', 'int')
        cur_keepdims = helper.get_var_attribute_by_name(node, 'keepdims', 'int')
        cur_axes = helper.get_list_attribute_by_name(node, 'axes', 'int')
        if pre_keepdims != 0 or cur_keepdims != 0:
            continue
        axes = sorted(pre_axes + cur_axes)
        if axes != [2, 3]:
            continue
        # Merge two ReduceMean into GlobalAveragePool.
        new_gap_node = onnx.helper.make_node(
            'GlobalAveragePool',
            [pre_node.input[0]],
            [node.output[0] + '_intermedia'],
            name = node.name + '_gap'
        )
        new_flatten_node = onnx.helper.make_node(
            'Flatten',
            [node.output[0] + '_intermedia'],
            [node.output[0]],
            name = node.name + '_flatten',
            axis = 1
        )

        # Clean up
        g.node.extend([new_gap_node, new_flatten_node])
        node_to_del.extend([pre_node, node])
        mid_val_info = helper.find_value_by_name(g, node.input[0])
        if mid_val_info:
            g.value_info.remove(mid_val_info)

    while node_to_del:
        node = node_to_del.pop()
        g.node.remove(node)

    topological_sort(g)

def fuse_slice_nodes_into_conv(g):
    # define pattern checker
    def check_is_slice(node):
        if node.op_type == 'Concat':
            return True
        if node.op_type != 'Slice':
            return False
        following_nodes = helper.find_following_nodes_by_input_value_name(g, node.output[0])
        if len(following_nodes) != 1:
            return False
        # also check attributes
        if len(node.input) != 5:
            return False
        # starts should be 0 or 1
        starts_node = helper.find_node_by_output_name(g, node.input[1])
        if starts_node.op_type != 'Constant':
            return False
        _, starts_list = helper.constant_to_list(starts_node)
        for num in starts_list:
            if num != 0 and num != 1:
                return False
        # ends
        ends_node = helper.find_node_by_output_name(g, node.input[2])
        if ends_node.op_type != 'Constant':
            return False
        # axes should be 2 or 3
        axes_node = helper.find_node_by_output_name(g, node.input[3])
        if axes_node.op_type != 'Constant':
            return False
        _, axes_list = helper.constant_to_list(axes_node)
        for num in axes_list:
            if num != 2 and num != 3:
                return False
        # Steps can only be 2
        steps_node = helper.find_node_by_output_name(g, node.input[4])
        if steps_node.op_type != 'Constant':
            return False
        _, steps_list = helper.constant_to_list(steps_node)
        for num in steps_list:
            if num != 2:
                return False
        # Recursion
        return check_is_slice(following_nodes[0])
    # defind concat finder
    def find_concat_node(node):
        while node.op_type != 'Concat':
            node = helper.find_following_nodes_by_input_value_name(g, node.output[0])[0]
        return node
    # define remove node function.
    def remove_nodes(input_name):
        following_nodes = helper.find_following_nodes_by_input_value_name(g, input_name)
        # Remove concat directly
        if len(following_nodes) == 1 and following_nodes[0].op_type == 'Concat':
            g.node.remove(following_nodes[0])
            return
        for following_node in following_nodes:
            # Recursion first
            remove_nodes(following_node.output[0])
            # Remove weights
            for i in range(1, len(following_node.input)):
                if len(helper.find_following_nodes_by_input_value_name(g, following_node.input[i])) > 1:
                    # More than one following nodes. Skip.
                    continue
                input_weight = helper.find_node_by_output_name(g, following_node.input[i])
                g.node.remove(input_weight)
            # Remove Slice nodes
            g.node.remove(following_node)
    # define remove value_info function
    def remove_value_infos(input_name):
        following_nodes = helper.find_following_nodes_by_input_value_name(g, input_name)
        if following_nodes[0].op_type == 'Concat':
            return
        for following_node in following_nodes:
            output_value = helper.find_value_by_name(g, following_node.output[0])
            # Remove output values
            if output_value is not None:
                g.value_info.remove(output_value)
            # Remove weight values
            for i in range(1, len(following_node.input)):
                input_value = helper.find_value_by_name(g, following_node.input[i])
                if input_value is not None:
                    g.value_info.remove(input_value)
            # Recursion
            remove_value_infos(following_node.output[0])
    # define get slice position
    def get_slice_position(final_slice_output):
        slice_position = [0, 0]
        prev_node = helper.find_node_by_output_name(g, final_slice_output)
        while prev_node is not None:
            starts_np = helper.constant_to_numpy(helper.find_node_by_output_name(g, prev_node.input[1]))
            axes_np = helper.constant_to_numpy(helper.find_node_by_output_name(g, prev_node.input[3]))
            for i in range(len(axes_np)):
                if axes_np[i] == 2:
                    slice_position[0] = starts_np[i]
                elif axes_np[i] == 3:
                    slice_position[1] = starts_np[i]
            prev_node = helper.find_node_by_output_name(g, prev_node.input[0])
        return slice_position
    # Check pattern from each input
    for input_value in g.input:
        nodes_after_input = helper.find_following_nodes_by_input_value_name(g, input_value.name)
        pattern_matched = True
        for following_node in nodes_after_input:
            if following_node.op_type != 'Slice':
                pattern_matched = False
                break
            else:
                pattern_matched = check_is_slice(following_node)
        if not pattern_matched:
            continue
        # Pattern found. Check limitation
        # Currently only support 2D
        if len(nodes_after_input) != 4:
            continue
        # Get the concat node
        concat_node = find_concat_node(nodes_after_input[0])
        # Get basic information
        input_shape = helper.get_shape_from_value_info(input_value)
        channel_num = input_shape[1]
        # Construct weight
        weight_np = np.zeros((input_shape[1] * 4, input_shape[1], 3, 3), dtype=np.float32)
        for i in range(4):
            # Check each branch
            slice_position = get_slice_position(concat_node.input[i])
            for j in range(channel_num):
                weight_np[i * channel_num + j, j, slice_position[0], slice_position[1]] = 1
        weight_node = helper.numpy_to_constant(concat_node.name + '_weight', weight_np)
        # Construct Conv node
        new_conv = onnx.helper.make_node(
            'Conv',
            [input_value.name, concat_node.name + '_weight'],
            [concat_node.output[0]],
            name = concat_node.name + '_fused',
            dilations = [1, 1],
            group = 1,
            kernel_shape = [3, 3],
            strides = [2, 2],
            pads = [0, 0, 2, 2]
        )
        # Delete old nodes, weights and value_infos
        remove_value_infos(input_value.name)
        remove_nodes(input_value.name)
        # Replace node
        g.node.append(weight_node)
        g.node.append(new_conv)
