import onnx
import numpy as np
from . import other
from . import helper


def expand_lstm_like_nodes(m):
    """Expand lstm node inside the model.
    """
    g = m.graph
    node_to_remove = []
    for node in g.node:
        if node.op_type == "LSTM":
            expand_LSTM_node(g, node)
            node_to_remove.append(node)
        elif node.op_type == "GRU":
            expand_GRU_node(g, node)
            node_to_remove.append(node)
        elif node.op_type == 'RNN':
            expand_RNN_node(g, node)
            node_to_remove.append(node)
        else:
            continue
    # After expansion, reinference the size. Later may introduce constant folding.
    for node in node_to_remove:
        g.node.remove(node)
    other.topological_sort(g)
    m = onnx.utils.polish_model(m)
    m = other.inference_shapes_until_complete_all(m)
    return m


def make_split(input_name: str, axis: int, split):
    """Make a split node.

    Args:
        input_name (str): input name of the split name.
        axis (int): axis to split on.
        split (List[int]): length of each output.

    Returns:
        List[str]: output names
        NodeProto: the generated split node.
    """
    output_names = [f"{input_name}_split_o{i}" for i in range(len(split))]
    split_node = onnx.helper.make_node(
        op_type='Split',
        inputs = [input_name],
        outputs = output_names,
        name = f"{input_name}_split",
        axis = axis,
        split = split
    )
    return output_names, split_node


def prepare_weight(w_iofc, bidirection=False):
    """Prepare weight for LSTM like nodes

    Args:
        w_iofc (str): input weight name
        bidirection (bool, optional): whether bidirectional. Defaults to False.

    Returns:
        str: forward weight name
        str: backward weight name
        List[NodeProto]: generate nodes
    """
    new_nodes = []
    if bidirection:
        two_names, split_node = make_split(w_iofc, 0, [1, 1])
        new_nodes.append(split_node)
        squeeze_names = [name + '_squeezed' for name in two_names]
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[0]],
            outputs = [squeeze_names[0]],
            name = squeeze_names[0],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[1]],
            outputs = [squeeze_names[1]],
            name = squeeze_names[1],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        return squeeze_names[0], squeeze_names[1], new_nodes
    else:
        squeeze_name = w_iofc + '_squeezed'
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [w_iofc],
            outputs = [squeeze_name],
            name = squeeze_name,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        return squeeze_name, '', new_nodes


def prepare_LSTM_bias(b_iofc, hidden_size, bidirection=False):
    """Prepare bias for LSTM like nodes

    Args:
        b_iofc (str): input bias name.
        hidden_size (int): hidden size.
        bidirection (bool, optional): whether bidirectional. Defaults to False.

    Returns:
        str: forward bias name
        str: backward bias name
        List[NodeProto]: generate nodes
    """
    new_nodes = []
    if bidirection:
        # Split direction
        two_names, split_node = make_split(b_iofc, 0, [1, 1])
        new_nodes.append(split_node)
        squeeze_names = [name + '_squeezed' for name in two_names]
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[0]],
            outputs = [squeeze_names[0]],
            name = squeeze_names[0],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[1]],
            outputs = [squeeze_names[1]],
            name = squeeze_names[1],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_names[0], 0, [4*hidden_size, 4*hidden_size])
        new_nodes.append(split_node)
        # Make add node
        add_name = b_iofc + '_Wb_Rb_sum'
        add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [add_name],
            name = add_name
        )
        new_nodes.append(add_node)
        # Split WB and RB
        wb_rb_names, split_node = make_split(squeeze_names[1], 0, [4*hidden_size, 4*hidden_size])
        new_nodes.append(split_node)
        # Make add node
        reverse_add_name = b_iofc + '_WB_RB_sum'
        reverse_add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [reverse_add_name],
            name = reverse_add_name
        )
        new_nodes.append(reverse_add_node)
        return add_name, reverse_add_name, new_nodes
    else:
        squeeze_name = b_iofc + '_squeezed'
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [b_iofc],
            outputs = [squeeze_name],
            name = squeeze_name,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_name, 0, [4*hidden_size, 4*hidden_size])
        new_nodes.append(split_node)
        # Make add node
        add_name = b_iofc + '_Wb_Rb_sum'
        add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [add_name],
            name = add_name
        )
        new_nodes.append(add_node)
        return add_name, '', new_nodes


def expand_LSTM_node(g, node):
    new_nodes = []
    # Get inputs
    input_x = node.input[0]
    input_w = node.input[1]
    input_r = node.input[2]
    input_b = ''
    input_sequence_lens = ''
    input_initial_h = ''
    input_initial_c = ''
    input_p = ''
    if len(node.input) > 3:
        input_b = node.input[3]
    if len(node.input) > 4:
        input_sequence_lens = node.input[4]
    if len(node.input) > 5:
        input_initial_h = node.input[5]
    if len(node.input) > 6:
        input_initial_c = node.input[6]
    if len(node.input) > 7:
        input_p = node.input[7]
    # Check inputs
    # Get info from input_x
    input_x_value_info = helper.find_value_by_name(g, input_x)
    if input_x_value_info is None:
        input_x_value_info = helper.find_input_by_name(g, input_x)
    if input_x_value_info is None:
        helper.logger.error(f"Cannot expand LSTM node {node.name}: cannot find input value_info {input_x}.")
        exit(1)
    input_x_shape = helper.get_shape_from_value_info(input_x_value_info)
    seq_length = input_x_shape[0]
    batch_size = input_x_shape[1]
    input_size = input_x_shape[2]
    # Get info from input_sequence_lens. It should be a constant
    if input_sequence_lens == '':
        sequence_lens = [seq_length] * batch_size
    else:
        input_sequence_lens_node = helper.find_node_by_output_name(g, input_sequence_lens)
        if input_sequence_lens_node is None:
            helper.logger.error(f"Cannot expand LSTM node {node.name}: input {input_sequence_lens} not found.")
            exit(1)
        if input_sequence_lens_node.op_type != "Constant":
            helper.logger.error(f"Cannot expand LSTM node {node.name}: input {input_sequence_lens} should be constant.")
            exit(1)
        sequence_lens_shape, sequence_lens = helper.constant_to_list(input_sequence_lens_node)
        if sequence_lens_shape[0] != batch_size:
            helper.logger.error(f"Cannot expand LSTM node {node.name}: input {input_sequence_lens} shape is invalid.")
            exit(1)
        for sequence_len in sequence_lens:
            if sequence_len != seq_length:
                helper.logger.error(f"Cannot expand LSTM node {node.name}: input {input_sequence_lens} currently is not supported. We only support same sequence lengths for now.")
                exit(1)
    # Get attributes
    activation_alpha = helper.get_list_attribute_by_name(node, 'activation_alpha', 'float')
    activation_beta = helper.get_list_attribute_by_name(node, 'activation_beta', 'float')
    activations = helper.get_list_attribute_by_name(node, 'activations', 'string')
    clip = helper.get_var_attribute_by_name(node, 'clip', 'float')
    direction = helper.get_var_attribute_by_name(node, 'direction', 'string')
    hidden_size = helper.get_var_attribute_by_name(node, 'hidden_size', 'int')
    input_forge = helper.get_var_attribute_by_name(node, 'input_forge', 'int')
    # Check attributes
    # TODO: support more attributes
    if activations is not None:
        helper.logger.error(f"Cannot expand LSTM node {node.name}: currently only support default activations. (Sigmoid, Tanh, Tanh)")
    if clip is not None:
        helper.logger.error(f"Cannot expand LSTM node {node.name}: currently do not support clip attribute.")
    if input_forge is not None and input_forge == 1:
        helper.logger.error(f"Cannot expand LSTM node {node.name}: currently do not support input forge.")
    if input_p != '':
         helper.logger.error(f"Cannot expand LSTM node {node.name}: currently do not support input p.")

    # Expand node into hidden cells (Single direction)
    if direction is None or direction == "forward" or direction == "reverse":
        x_list = []
        y_h_list = []
        y_c_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if direction == "reverse":
            x_list.reverse()
        w_name, _, generated_nodes = prepare_weight(input_w)
        new_nodes.extend(generated_nodes)
        r_name, _, generated_nodes = prepare_weight(input_r)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_name, _, generated_nodes = prepare_LSTM_bias(input_b, hidden_size)
            new_nodes.extend(generated_nodes)
        else:
            b_name = input_b
        if input_p != '':
            p_name, _, generated_nodes = prepare_weight(input_p)
            new_nodes.extend(generated_nodes)
        else:
            p_name = input_p
        # Expand lstm block
        h_pre = input_initial_h
        c_pre = input_initial_c
        for x in x_list:
            y_h_name = x + '_out_y_h'
            y_c_name = x + '_out_y_c'
            generated_nodes = make_LSTM_block(x, w_name, r_name, b_name, h_pre, c_pre, p_name, y_h_name, y_c_name, hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            y_c_list.append(y_c_name)
            h_pre = y_h_name
            c_pre = y_c_name
        # Prepare the output y
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            if direction == "reverse":
                y_h_unsqueezed_output_list.reverse()
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h and y_c if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)
        if len(node.output) > 2 and node.output[2] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [y_c_list[-1]],
                outputs = [node.output[2]],
                name = node.name + '_Y_c_identity'
            )
            new_nodes.append(identity_node)

    # Expand node into hidden cells (Bidirection)
    elif direction == "bidirectional":
        x_list = []
        y_h_list = []
        y_c_list = []
        reversed_y_h_list = []
        reversed_y_c_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if input_initial_h == '':
            initial_h_list = ['', '']
        else:
            initial_h_list, split_node = make_split(input_initial_h, 0, [1, 1])
            new_nodes.append(split_node)
        if input_initial_c == '':
            initial_c_list = ['', '']
        else:
            initial_c_list, split_node = make_split(input_initial_c, 0, [1, 1])
            new_nodes.append(split_node)
        w_name, reverse_w_name, generated_nodes = prepare_weight(input_w, True)
        new_nodes.extend(generated_nodes)
        r_name, reverse_r_name, generated_nodes = prepare_weight(input_r, True)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_name, reverse_b_name, generated_nodes = prepare_LSTM_bias(input_b, hidden_size, True)
            new_nodes.extend(generated_nodes)
        else:
            b_name = input_b
            reverse_b_name = ''
        if input_p != '':
            p_name, reverse_p_name, generated_nodes = prepare_weight(input_p, True)
            new_nodes.extend(generated_nodes)
        else:
            p_name = input_p
            reverse_p_name = ''
        # Expand lstm block (forward)
        h_pre = initial_h_list[0]
        c_pre = initial_c_list[0]
        for x in x_list:
            y_h_name = x + '_out_y_h'
            y_c_name = x + '_out_y_c'
            generated_nodes = make_LSTM_block(x, w_name, r_name, b_name, h_pre, c_pre, p_name, y_h_name, y_c_name, hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            y_c_list.append(y_c_name)
            h_pre = y_h_name
            c_pre = y_c_name
        # Expand lstm block (reverse)
        h_pre = initial_h_list[1]
        c_pre = initial_c_list[1]
        for x in reversed(x_list):
            y_h_name = x + '_rout_y_h'
            y_c_name = x + '_rout_y_c'
            x_id_name = x + '_identity'
            identity_node = onnx.helper.make_node("Identity", [x], [x_id_name], name=x_id_name)
            new_nodes.append(identity_node)
            generated_nodes = make_LSTM_block(x_id_name, reverse_w_name, reverse_r_name, reverse_b_name, h_pre, c_pre, reverse_p_name, y_h_name, y_c_name,
                hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            reversed_y_h_list.append(y_h_name)
            reversed_y_c_list.append(y_c_name)
            h_pre = y_h_name
            c_pre = y_c_name
        reversed_y_h_list.reverse()
        reversed_y_c_list.reverse()
        # Prepare the output y
        final_y_h_list = []
        for i in range(seq_length):
            final_y_h_name = x_list[i] + '_concat_y_h'
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = [y_h_list[i], reversed_y_h_list[i]],
                outputs = [final_y_h_name],
                name = final_y_h_name,
                axis = 0
            )
            new_nodes.append(concat_node)
            final_y_h_list.append(final_y_h_name)
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in final_y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [final_y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h and y_c if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [final_y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)
        if len(node.output) > 2 and node.output[2] != '':
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = [y_c_list[-1], reversed_y_c_list[-1]],
                outputs = [node.output[2]],
                name = x_list[-1] + '_concat_y_c',
                axis = 0
            )
            new_nodes.append(concat_node)
    else:
        helper.logger.error(f"Cannot expand LSTM node {node.name}: invalid direction {direction}.")
        exit(1)
    g.node.extend(new_nodes)


def make_LSTM_block(x_t, w_iofc, r_iofc, b_iofc, h_pre, c_pre, p_iof, y_h, y_c, hidden_size, batch_size):
    new_nodes = []
    # Squeeze x. The output shape should be [batch, input]
    x_t_squeezed = x_t + '_squeezed'
    squeeze_node = onnx.helper.make_node(
        op_type = "Squeeze",
        inputs = [x_t],
        outputs = [x_t_squeezed],
        name = x_t_squeezed,
        axes = [0]
    )
    new_nodes.append(squeeze_node)
    # Make Gemm for Xt*(W^T) + Wb + Rb. [batch, 4*hidden]
    gemm_0_name = x_t + '_gemm_0'
    if b_iofc == '':
        gemm_0_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = [x_t_squeezed, w_iofc],
            outputs = [gemm_0_name],
            name = gemm_0_name,
            transB = 1
        )
    else:
        gemm_0_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = [x_t_squeezed, w_iofc, b_iofc],
            outputs = [gemm_0_name],
            name = gemm_0_name,
            transB = 1
        )
    new_nodes.append(gemm_0_node)
    # Make Gemm for Ht-1*(R^T) [batch, 4*hidden]
    h_prev_squeezed = x_t + '_ht-1'
    if h_pre == '':
        # No previous H found, make 0 constant node.
        h_prev_np = np.zeros((batch_size, hidden_size), dtype='float32')
        constant_node = helper.numpy_to_constant(h_prev_squeezed, h_prev_np)
        new_nodes.append(constant_node)
    else:
        # Squeeze h
        squeeze_node = onnx.helper.make_node(
            op_type = "Squeeze",
            inputs = [h_pre],
            outputs = [h_prev_squeezed],
            name = h_prev_squeezed,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
    gemm_1_name = x_t + '_gemm_1'
    gemm_1_node = onnx.helper.make_node(
        op_type = 'Gemm',
        inputs = [h_prev_squeezed, r_iofc],
        outputs = [gemm_1_name],
        name = gemm_1_name,
        transB = 1
    )
    new_nodes.append(gemm_1_node)
    # Split iofc [batch, hidden]
    gemm_0_output_splitted, split_node = make_split(gemm_0_name, 1, [hidden_size] * 4)
    new_nodes.append(split_node)
    gemm_1_output_splitted, split_node = make_split(gemm_1_name, 1, [hidden_size] * 4)
    new_nodes.append(split_node)
    # Prepare it [batch, hidden]
    it_add_name = x_t + '_it_add'
    it_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [gemm_0_output_splitted[0], gemm_1_output_splitted[0]],
        outputs = [it_add_name],
        name = it_add_name
    )
    new_nodes.append(it_add_node)
    it_f_name = x_t + '_it_f'
    it_f_node = onnx.helper.make_node(
        op_type = "Sigmoid",
        inputs = [it_add_name],
        outputs = [it_f_name],
        name = it_f_name
    )
    new_nodes.append(it_f_node)
    # Prepare ft
    ft_add_name = x_t + '_ft_add'
    ft_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [gemm_0_output_splitted[2], gemm_1_output_splitted[2]],
        outputs = [ft_add_name],
        name = ft_add_name
    )
    new_nodes.append(ft_add_node)
    ft_f_name = x_t + '_ft_f'
    ft_f_node = onnx.helper.make_node(
        op_type = "Sigmoid",
        inputs = [ft_add_name],
        outputs = [ft_f_name],
        name = ft_f_name
    )
    new_nodes.append(ft_f_node)
    # Prepare ct
    ct_add_name = x_t + '_ct_add'
    ct_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [gemm_0_output_splitted[3], gemm_1_output_splitted[3]],
        outputs = [ct_add_name],
        name = ct_add_name
    )
    new_nodes.append(ct_add_node)
    ct_g_name = x_t + '_ct_g'
    ct_g_node = onnx.helper.make_node(
        op_type = "Tanh",
        inputs = [ct_add_name],
        outputs = [ct_g_name],
        name = ct_g_name
    )
    new_nodes.append(ct_g_node)
    # Prepare y_c
    c_prev_squeezed = x_t + '_ct-1'
    if c_pre == '':
        # No previous H found, make 0 constant node.
        c_prev_np = np.zeros((batch_size, hidden_size), dtype='float32')
        constant_node = helper.numpy_to_constant(c_prev_squeezed, c_prev_np)
        new_nodes.append(constant_node)
    else:
        # Squeeze h
        squeeze_node = onnx.helper.make_node(
            op_type = "Squeeze",
            inputs = [c_pre],
            outputs = [c_prev_squeezed],
            name = c_prev_squeezed,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
    y_c_mul_0_name = x_t + '_y_c_mul_0'
    y_c_mul_0_node = onnx.helper.make_node(
        op_type = 'Mul',
        inputs = [ft_f_name, c_prev_squeezed],
        outputs = [y_c_mul_0_name],
        name = y_c_mul_0_name
    )
    new_nodes.append(y_c_mul_0_node)
    y_c_mul_1_name = x_t + '_y_c_mul_1'
    y_c_mul_1_node = onnx.helper.make_node(
        op_type = 'Mul',
        inputs = [it_f_name, ct_g_name],
        outputs = [y_c_mul_1_name],
        name = y_c_mul_1_name
    )
    new_nodes.append(y_c_mul_1_node)
    y_c_add_name = x_t + '_y_c_add'
    y_c_add_node = onnx.helper.make_node(
        op_type = 'Add',
        inputs = [y_c_mul_0_name, y_c_mul_1_name],
        outputs = [y_c_add_name],
        name = y_c_add_name
    )
    new_nodes.append(y_c_add_node)
    y_c_unsqueeze_node = onnx.helper.make_node(
        op_type = 'Unsqueeze',
        inputs = [y_c_add_name],
        outputs = [y_c],
        name = y_c,
        axes = [0]
    )
    new_nodes.append(y_c_unsqueeze_node)
    # Prepare ot
    ot_add_name = x_t + '_ot_add'
    ot_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [gemm_0_output_splitted[1], gemm_1_output_splitted[1]],
        outputs = [ot_add_name],
        name = ot_add_name
    )
    new_nodes.append(ot_add_node)
    ot_f_name = x_t + '_ot_f'
    ot_f_node = onnx.helper.make_node(
        op_type = "Sigmoid",
        inputs = [ot_add_name],
        outputs = [ot_f_name],
        name = ot_f_name
    )
    new_nodes.append(ot_f_node)
    # Prepare y_h
    y_h_h_name = x_t + '_y_h_h'
    y_h_h_node = onnx.helper.make_node(
        op_type = "Tanh",
        inputs = [y_c],
        outputs = [y_h_h_name],
        name = y_h_h_name
    )
    new_nodes.append(y_h_h_node)
    y_h_mul_node = onnx.helper.make_node(
        op_type = "Mul",
        inputs = [ot_f_name, y_h_h_name],
        outputs = [y_h],
        name = y_h
    )
    new_nodes.append(y_h_mul_node)
    return new_nodes


def prepare_GRU_bias(b_zrh, hidden_size, bidirection=False):
    """Prepare bias for LSTM like nodes

    Args:
        b_zrh (str): input bias name.
        hidden_size (int): hidden size.
        bidirection (bool, optional): whether bidirectional. Defaults to False.

    Returns:
        str: forward bias name
        str: backward bias name
        List[NodeProto]: generate nodes
    """
    new_nodes = []
    if bidirection:
        # Split direction
        two_names, split_node = make_split(b_zrh, 0, [1, 1])
        new_nodes.append(split_node)
        squeeze_names = [name + '_squeezed' for name in two_names]
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[0]],
            outputs = [squeeze_names[0]],
            name = squeeze_names[0],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[1]],
            outputs = [squeeze_names[1]],
            name = squeeze_names[1],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_names[0], 0, [hidden_size * 3, hidden_size * 3])
        new_nodes.append(split_node)
        # Split WB and RB
        wB_rB_names, split_node = make_split(squeeze_names[1], 0, [hidden_size * 3, hidden_size * 3])
        new_nodes.append(split_node)
        return wb_rb_names, wB_rB_names, new_nodes
    else:
        squeeze_name = b_zrh + '_squeezed'
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [b_zrh],
            outputs = [squeeze_name],
            name = squeeze_name,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_name, 0, [hidden_size * 3, hidden_size * 3])
        new_nodes.append(split_node)
        return wb_rb_names, ['', ''], new_nodes


def expand_GRU_node(g, node):
    new_nodes = []
    # Get inputs
    input_x = node.input[0]
    input_w = node.input[1]
    input_r = node.input[2]
    input_b = ''
    input_sequence_lens = ''
    input_initial_h = ''
    if len(node.input) > 3:
        input_b = node.input[3]
    if len(node.input) > 4:
        input_sequence_lens = node.input[4]
    if len(node.input) > 5:
        input_initial_h = node.input[5]
    # Check inputs
    # Get info from input_x
    input_x_value_info = helper.find_value_by_name(g, input_x)
    if input_x_value_info is None:
        input_x_value_info = helper.find_input_by_name(g, input_x)
    if input_x_value_info is None:
        helper.logger.error(f"Cannot expand GRU node {node.name}: cannot find input value_info {input_x}.")
        exit(1)
    input_x_shape = helper.get_shape_from_value_info(input_x_value_info)
    seq_length = input_x_shape[0]
    batch_size = input_x_shape[1]
    input_size = input_x_shape[2]
    # Get info from input_sequence_lens. It should be a constant
    if input_sequence_lens == '':
        sequence_lens = [seq_length] * batch_size
    else:
        input_sequence_lens_node = helper.find_node_by_output_name(g, input_sequence_lens)
        if input_sequence_lens_node is None:
            helper.logger.error(f"Cannot expand GRU node {node.name}: input {input_sequence_lens} not found.")
            exit(1)
        if input_sequence_lens_node.op_type != "Constant":
            helper.logger.error(f"Cannot expand GRU node {node.name}: input {input_sequence_lens} should be constant.")
            exit(1)
        sequence_lens_shape, sequence_lens = helper.constant_to_list(input_sequence_lens_node)
        if sequence_lens_shape[0] != batch_size:
            helper.logger.error(f"Cannot expand GRU node {node.name}: input {input_sequence_lens} shape is invalid.")
            exit(1)
        for sequence_len in sequence_lens:
            if sequence_len != seq_length:
                helper.logger.error(f"Cannot expand GRU node {node.name}: input {input_sequence_lens} currently is not supported. We only support same sequence lengths for now.")
                exit(1)
    # Get attributes
    activation_alpha = helper.get_list_attribute_by_name(node, 'activation_alpha', 'float')
    activation_beta = helper.get_list_attribute_by_name(node, 'activation_beta', 'float')
    activations = helper.get_list_attribute_by_name(node, 'activations', 'string')
    clip = helper.get_var_attribute_by_name(node, 'clip', 'float')
    direction = helper.get_var_attribute_by_name(node, 'direction', 'string')
    hidden_size = helper.get_var_attribute_by_name(node, 'hidden_size', 'int')
    linear_before_reset = helper.get_var_attribute_by_name(node, 'linear_before_reset', 'int')
    # Check attributes
    # TODO: support more attributes
    if activations is not None:
        helper.logger.error(f"Cannot expand GRU node {node.name}: currently only support default activations. (Sigmoid, Tanh)")
    if clip is not None:
        helper.logger.error(f"Cannot expand GRU node {node.name}: currently do not support clip attribute.")

    # Expand node into hidden cells (Single direction)
    if direction is None or direction == "forward" or direction == "reverse":
        x_list = []
        y_h_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if direction == "reverse":
            x_list.reverse()
        w_name, _, generated_nodes = prepare_weight(input_w)
        new_nodes.extend(generated_nodes)
        r_name, _, generated_nodes = prepare_weight(input_r)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_names, _, generated_nodes = prepare_GRU_bias(input_b, hidden_size)
            new_nodes.extend(generated_nodes)
        else:
            b_names = [''] * 2
        # Expand gru block
        h_pre = input_initial_h
        for x in x_list:
            y_h_name = x + '_out_y_h'
            generated_nodes = make_GRU_block(x, w_name, r_name, b_names, h_pre, y_h_name, hidden_size, batch_size, linear_before_reset)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            h_pre = y_h_name
        # Prepare the output y
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            if direction == "reverse":
                y_h_unsqueezed_output_list.reverse()
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)

    # Expand node into hidden cells (Bidirection)
    elif direction == "bidirectional":
        x_list = []
        y_h_list = []
        reversed_y_h_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if input_initial_h == '':
            initial_h_list = ['', '']
        else:
            initial_h_list, split_node = make_split(input_initial_h, 0, [1, 1])
            new_nodes.append(split_node)
        w_name, reverse_w_name, generated_nodes = prepare_weight(input_w, True)
        new_nodes.extend(generated_nodes)
        r_name, reverse_r_name, generated_nodes = prepare_weight(input_r, True)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_names, reverse_b_names, generated_nodes = prepare_GRU_bias(input_b, hidden_size, True)
            new_nodes.extend(generated_nodes)
        else:
            b_names = [''] * 2
            reverse_b_names = b_names
        # Expand GRU block (forward)
        h_pre = initial_h_list[0]
        for x in x_list:
            y_h_name = x + '_out_y_h'
            generated_nodes = make_GRU_block(x, w_name, r_name, b_names, h_pre, y_h_name, hidden_size, batch_size, linear_before_reset)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            h_pre = y_h_name
        # Expand gru block (reverse)
        h_pre = initial_h_list[1]
        for x in reversed(x_list):
            y_h_name = x + '_rout_y_h'
            x_id_name = x + '_identity'
            identity_node = onnx.helper.make_node("Identity", [x], [x_id_name], name=x_id_name)
            new_nodes.append(identity_node)
            generated_nodes = make_GRU_block(x_id_name, reverse_w_name, reverse_r_name, reverse_b_names, h_pre, y_h_name,
                hidden_size, batch_size, linear_before_reset)
            new_nodes.extend(generated_nodes)
            reversed_y_h_list.append(y_h_name)
            h_pre = y_h_name
        reversed_y_h_list.reverse()
        # Prepare the output y
        final_y_h_list = []
        for i in range(seq_length):
            final_y_h_name = x_list[i] + '_concat_y_h'
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = [y_h_list[i], reversed_y_h_list[i]],
                outputs = [final_y_h_name],
                name = final_y_h_name,
                axis = 0
            )
            new_nodes.append(concat_node)
            final_y_h_list.append(final_y_h_name)
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in final_y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [final_y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [final_y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)
    else:
        helper.logger.error(f"Cannot expand GRU node {node.name}: invalid direction {direction}.")
        exit(1)
    g.node.extend(new_nodes)


def make_GRU_block(x_t, w_zrh, r_zrh, b_names, h_pre, y_h_name, hidden_size, batch_size, linear_before_reset):
    new_nodes = []
    # Squeeze x. The output shape should be [batch, input]
    x_t_squeezed = x_t + '_squeezed'
    squeeze_node = onnx.helper.make_node(
        op_type = "Squeeze",
        inputs = [x_t],
        outputs = [x_t_squeezed],
        name = x_t_squeezed,
        axes = [0]
    )
    new_nodes.append(squeeze_node)
    # Make Gemm for Xt*(W^T) + Wb. [batch, 3*hidden]
    gemm_0_name = x_t + '_gemm_0'
    inputs = [x_t_squeezed, w_zrh]
    if b_names[0] != '':
        inputs.append(b_names[0])
    gemm_0_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [gemm_0_name],
            name = gemm_0_name,
            transB = 1
    )
    new_nodes.append(gemm_0_node)
    # Split zrh [batch, hidden]
    gemm_0_output_splitted, split_node = make_split(gemm_0_name, 1, [hidden_size] * 3)
    new_nodes.append(split_node)
    # Make Gemm for Ht-1*(R^T) [batch, 4*hidden]
    h_prev_squeezed = x_t + '_ht-1'
    if h_pre == '':
        # No previous H found, make 0 constant node.
        h_prev_np = np.zeros((batch_size, hidden_size), dtype='float32')
        constant_node = helper.numpy_to_constant(h_prev_squeezed, h_prev_np)
        new_nodes.append(constant_node)
    else:
        # Squeeze h
        squeeze_node = onnx.helper.make_node(
            op_type = "Squeeze",
            inputs = [h_pre],
            outputs = [h_prev_squeezed],
            name = h_prev_squeezed,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
    if linear_before_reset != 0:
        # ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        gemm_1_name = x_t + '_gemm_1'
        inputs = [h_prev_squeezed, r_zrh]
        if b_names[1] != '':
            inputs.append(b_names[1])
        gemm_1_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [gemm_1_name],
            name = gemm_1_name,
            transB = 1
        )
        new_nodes.append(gemm_1_node)
        # Split zrh
        gemm_1_output_splitted, split_node = make_split(gemm_1_name, 1, [hidden_size] * 3)
        new_nodes.append(split_node)
        # Prepare zt [batch, hidden]
        zt_add_name = x_t + '_zt_add'
        zt_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[0], gemm_1_output_splitted[0]],
            outputs = [zt_add_name],
            name = zt_add_name
        )
        new_nodes.append(zt_add_node)
        zt_f_name = x_t + '_zt_f'
        zt_f_node = onnx.helper.make_node(
            op_type = "Sigmoid",
            inputs = [zt_add_name],
            outputs = [zt_f_name],
            name = zt_f_name
        )
        new_nodes.append(zt_f_node)
        # Prepare rt
        rt_add_name = x_t + '_rt_add'
        rt_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[1], gemm_1_output_splitted[1]],
            outputs = [rt_add_name],
            name = rt_add_name
        )
        new_nodes.append(rt_add_node)
        rt_f_name = x_t + '_rt_f'
        rt_f_node = onnx.helper.make_node(
            op_type = "Sigmoid",
            inputs = [rt_add_name],
            outputs = [rt_f_name],
            name = rt_f_name
        )
        new_nodes.append(rt_f_node)
        # Prepare ht
        ht_mul_name = x_t + '_ht_mul'
        ht_mul_node =  onnx.helper.make_node(
            op_type = "Mul",
            inputs = [rt_f_name, gemm_1_output_splitted[2]],
            outputs = [ht_mul_name],
            name = ht_mul_name
        )
        new_nodes.append(ht_mul_node)
        ht_add_name = x_t + '_ht_add'
        ht_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[2], ht_mul_name],
            outputs = [ht_add_name],
            name = ht_add_name
        )
        new_nodes.append(ht_add_node)
        ht_g_name = x_t + '_ht_g'
        ht_g_node = onnx.helper.make_node(
            op_type = "Tanh",
            inputs = [ht_add_name],
            outputs = [ht_g_name],
            name = ht_g_name
        )
        new_nodes.append(ht_g_node)
    else:
        # ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        # Split R and B first
        r_splitted, generated_nodes = make_split(r_zrh, 0, [hidden_size] * 3)
        new_nodes.extend(generated_nodes)
        if b_names[1] != '':
            b_splitted, generated_nodes = make_split(b_names[1], 0, [hidden_size] * 3)
            new_nodes.extend(generated_nodes)
        else:
            b_splitted = [''] * 3
        # Prepare zt [batch, hidden]
        zt_gemm_name = x_t + '_zt_gemm'
        inputs = [h_prev_squeezed, r_splitted[0]]
        if b_splitted[0] != '':
            inputs.append(b_splitted[0])
        zt_gemm_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [zt_gemm_name],
            name = zt_gemm_name,
            transB = 1
        )
        new_nodes.append(zt_gemm_node)
        zt_add_name = x_t + '_zt_add'
        zt_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[0], zt_gemm_name],
            outputs = [zt_add_name],
            name = zt_add_name
        )
        new_nodes.append(zt_add_node)
        zt_f_name = x_t + '_zt_f'
        zt_f_node = onnx.helper.make_node(
            op_type = "Sigmoid",
            inputs = [zt_add_name],
            outputs = [zt_f_name],
            name = zt_f_name
        )
        new_nodes.append(zt_f_node)
        # Prepare rt
        rt_gemm_name = x_t + '_rt_gemm'
        inputs = [h_prev_squeezed, r_splitted[1]]
        if b_splitted[1] != '':
            inputs.append(b_splitted[1])
        rt_gemm_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [rt_gemm_name],
            name = rt_gemm_name,
            transB = 1
        )
        new_nodes.append(rt_gemm_node)
        rt_add_name = x_t + '_rt_add'
        rt_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[1], rt_gemm_name],
            outputs = [rt_add_name],
            name = rt_add_name
        )
        new_nodes.append(rt_add_node)
        rt_f_name = x_t + '_rt_f'
        rt_f_node = onnx.helper.make_node(
            op_type = "Sigmoid",
            inputs = [rt_add_name],
            outputs = [rt_f_name],
            name = rt_f_name
        )
        new_nodes.append(rt_f_node)
        # Prepare ht
        ht_mul_name = x_t + '_ht_mul'
        ht_mul_node =  onnx.helper.make_node(
            op_type = "Mul",
            inputs = [rt_f_name, h_prev_squeezed],
            outputs = [ht_mul_name],
            name = ht_mul_name
        )
        new_nodes.append(ht_mul_node)
        ht_gemm_name = x_t + '_ht_gemm'
        inputs = [ht_mul_name, r_splitted[2]]
        if b_splitted[2] != '':
            inputs.append(b_splitted[2])
        ht_gemm_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [ht_gemm_name],
            name = ht_gemm_name,
            transB = 1
        )
        new_nodes.append(ht_gemm_node)
        ht_add_name = x_t + '_ht_add'
        ht_add_node = onnx.helper.make_node(
            op_type = "Add",
            inputs = [gemm_0_output_splitted[2], ht_gemm_name],
            outputs = [ht_add_name],
            name = ht_add_name
        )
        new_nodes.append(ht_add_node)
        ht_g_name = x_t + '_ht_g'
        ht_g_node = onnx.helper.make_node(
            op_type = "Tanh",
            inputs = [ht_add_name],
            outputs = [ht_g_name],
            name = ht_g_name
        )
        new_nodes.append(ht_g_node)
    # Prepare y_h
    y_h_unsqueeze_name = x_t + '_y_h_unsqueeze'
    y_h_unsqueeze_node = onnx.helper.make_node(
        op_type = "Unsqueeze",
        inputs = [zt_f_name],
        outputs = [y_h_unsqueeze_name],
        name = y_h_unsqueeze_name,
        axes = [0]
    )
    new_nodes.append(y_h_unsqueeze_node)
    # Construct BN node
    y_h_bn_name = x_t + '_y_h_bn'
    minus_ones = [-1.0] * batch_size
    ones = [1.0] * batch_size
    zeros = [0.0] * batch_size
    scale_node = helper.list_to_constant(y_h_bn_name + "_scale", [batch_size], minus_ones)
    bias_node = helper.list_to_constant(y_h_bn_name + "_bias", [batch_size], ones)
    mean_node = helper.list_to_constant(y_h_bn_name + "_mean", [batch_size], zeros)
    var_node = helper.list_to_constant(y_h_bn_name + "_var", [batch_size], ones)
    y_h_bn_node = onnx.helper.make_node(
        op_type = "BatchNormalization",
        inputs = [y_h_unsqueeze_name,
        scale_node.output[0],
        bias_node.output[0],
        mean_node.output[0],
        var_node.output[0]],
        outputs = [y_h_bn_name],
        name = y_h_bn_name
    )
    new_nodes.extend([scale_node, bias_node, mean_node, var_node, y_h_bn_node])
    y_h_mul_0_name = x_t + '_y_h_mul_0'
    y_h_mul_0_node = onnx.helper.make_node(
        op_type = "Mul",
        inputs = [y_h_bn_name, ht_g_name],
        outputs = [y_h_mul_0_name],
        name = y_h_mul_0_name
    )
    new_nodes.append(y_h_mul_0_node)
    y_h_mul_1_name = x_t + '_y_h_mul_1'
    y_h_mul_1_node = onnx.helper.make_node(
        op_type = "Mul",
        inputs = [y_h_unsqueeze_name, h_pre],
        outputs = [y_h_mul_1_name],
        name = y_h_mul_1_name
    )
    new_nodes.append(y_h_mul_1_node)
    y_h_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [y_h_mul_0_name, y_h_mul_1_name],
        outputs = [y_h_name],
        name = y_h_name
    )
    new_nodes.append(y_h_add_node)
    return new_nodes

def prepare_RNN_bias(b_i, hidden_size, bidirection=False):
    """Prepare bias for RNN nodes

    Args:
        b_i (str): input bias name.
        hidden_size (int): hidden size.
        bidirection (bool, optional): whether bidirectional. Defaults to False.

    Returns:
        str: forward bias name
        str: backward bias name
        List[NodeProto]: generate nodes
    """
    new_nodes = []
    if bidirection:
        # Split direction
        two_names, split_node = make_split(b_i, 0, [1, 1])
        new_nodes.append(split_node)
        squeeze_names = [name + '_squeezed' for name in two_names]
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[0]],
            outputs = [squeeze_names[0]],
            name = squeeze_names[0],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [two_names[1]],
            outputs = [squeeze_names[1]],
            name = squeeze_names[1],
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_names[0], 0, [hidden_size, hidden_size])
        new_nodes.append(split_node)
        # Make add node
        add_name = b_i + '_Wb_Rb_sum'
        add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [add_name],
            name = add_name
        )
        new_nodes.append(add_node)
        # Split WB and RB
        wb_rb_names, split_node = make_split(squeeze_names[1], 0, [hidden_size, hidden_size])
        new_nodes.append(split_node)
        # Make add node
        reverse_add_name = b_i + '_WB_RB_sum'
        reverse_add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [reverse_add_name],
            name = reverse_add_name
        )
        new_nodes.append(reverse_add_node)
        return add_name, reverse_add_name, new_nodes
    else:
        squeeze_name = b_i + '_squeezed'
        squeeze_node = onnx.helper.make_node(
            op_type = 'Squeeze',
            inputs = [b_i],
            outputs = [squeeze_name],
            name = squeeze_name,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
        # Split Wb and Rb
        wb_rb_names, split_node = make_split(squeeze_name, 0, [hidden_size, hidden_size])
        new_nodes.append(split_node)
        # Make add node
        add_name = b_i + '_Wb_Rb_sum'
        add_node = onnx.helper.make_node(
            op_type = 'Add',
            inputs = wb_rb_names,
            outputs = [add_name],
            name = add_name
        )
        new_nodes.append(add_node)
        return add_name, '', new_nodes


def expand_RNN_node(g, node):
    new_nodes = []
    # Get inputs
    input_x = node.input[0]
    input_w = node.input[1]
    input_r = node.input[2]
    input_b = ''
    input_sequence_lens = ''
    input_initial_h = ''
    if len(node.input) > 3:
        input_b = node.input[3]
    if len(node.input) > 4:
        input_sequence_lens = node.input[4]
    if len(node.input) > 5:
        input_initial_h = node.input[5]
    # Check inputs
    # Get info from input_x
    input_x_value_info = helper.find_value_by_name(g, input_x)
    if input_x_value_info is None:
        input_x_value_info = helper.find_input_by_name(g, input_x)
    if input_x_value_info is None:
        helper.logger.error(f"Cannot expand RNN node {node.name}: cannot find input value_info {input_x}.")
        exit(1)
    input_x_shape = helper.get_shape_from_value_info(input_x_value_info)
    seq_length = input_x_shape[0]
    batch_size = input_x_shape[1]
    input_size = input_x_shape[2]
    # Get info from input_sequence_lens. It should be a constant
    if input_sequence_lens == '':
        sequence_lens = [seq_length] * batch_size
    else:
        input_sequence_lens_node = helper.find_node_by_output_name(g, input_sequence_lens)
        if input_sequence_lens_node is None:
            helper.logger.error(f"Cannot expand RNN node {node.name}: input {input_sequence_lens} not found.")
            exit(1)
        if input_sequence_lens_node.op_type != "Constant":
            helper.logger.error(f"Cannot expand RNN node {node.name}: input {input_sequence_lens} should be constant.")
            exit(1)
        sequence_lens_shape, sequence_lens = helper.constant_to_list(input_sequence_lens_node)
        if sequence_lens_shape[0] != batch_size:
            helper.logger.error(f"Cannot expand RNN node {node.name}: input {input_sequence_lens} shape is invalid.")
            exit(1)
        for sequence_len in sequence_lens:
            if sequence_len != seq_length:
                helper.logger.error(f"Cannot expand RNN node {node.name}: input {input_sequence_lens} currently is not supported. We only support same sequence lengths for now.")
                exit(1)
    # Get attributes
    activation_alpha = helper.get_list_attribute_by_name(node, 'activation_alpha', 'float')
    activation_beta = helper.get_list_attribute_by_name(node, 'activation_beta', 'float')
    activations = helper.get_list_attribute_by_name(node, 'activations', 'string')
    clip = helper.get_var_attribute_by_name(node, 'clip', 'float')
    direction = helper.get_var_attribute_by_name(node, 'direction', 'string')
    hidden_size = helper.get_var_attribute_by_name(node, 'hidden_size', 'int')
    # Check attributes
    # TODO: support more attributes
    if activations is not None and activations != [b'Tanh'] and activations != [b'Tanh', b'Tanh']:
        helper.logger.error(f"Cannot expand RNN node {node.name}: currently only support default activations. (Tanh) or (Tanh, Tanh)")
    if clip is not None:
        helper.logger.error(f"Cannot expand RNN node {node.name}: currently do not support clip attribute.")

    # Expand node into hidden cells (Single direction)
    if direction is None or direction == "forward" or direction == "reverse":
        x_list = []
        y_h_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if direction == "reverse":
            x_list.reverse()
        w_name, _, generated_nodes = prepare_weight(input_w)
        new_nodes.extend(generated_nodes)
        r_name, _, generated_nodes = prepare_weight(input_r)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_name, _, generated_nodes = prepare_RNN_bias(input_b, hidden_size)
            new_nodes.extend(generated_nodes)
        else:
            b_name = ''
        # Expand rnn block
        h_pre = input_initial_h
        for x in x_list:
            y_h_name = x + '_out_y_h'
            generated_nodes = make_RNN_block(x, w_name, r_name, b_name, h_pre, y_h_name, hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            h_pre = y_h_name
        # Prepare the output y
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            if direction == "reverse":
                y_h_unsqueezed_output_list.reverse()
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)

    # Expand node into hidden cells (Bidirection)
    elif direction == "bidirectional":
        x_list = []
        y_h_list = []
        reversed_y_h_list = []
        # Prepare inputs
        if seq_length != 1:
            # Split input.
            x_list, split_node = make_split(input_x, 0, [1] * seq_length)
            new_nodes.append(split_node)
        else:
            # No need for split the input
            x_list.append(input_x)
        if input_initial_h == '':
            initial_h_list = ['', '']
        else:
            initial_h_list, split_node = make_split(input_initial_h, 0, [1, 1])
            new_nodes.append(split_node)
        w_name, reverse_w_name, generated_nodes = prepare_weight(input_w, True)
        new_nodes.extend(generated_nodes)
        r_name, reverse_r_name, generated_nodes = prepare_weight(input_r, True)
        new_nodes.extend(generated_nodes)
        if input_b != '':
            b_name, reverse_b_name, generated_nodes = prepare_RNN_bias(input_b, hidden_size, True)
            new_nodes.extend(generated_nodes)
        else:
            b_name = ''
            reverse_b_name = b_name
        # Expand RNN block (forward)
        h_pre = initial_h_list[0]
        for x in x_list:
            y_h_name = x + '_out_y_h'
            generated_nodes = make_RNN_block(x, w_name, r_name, b_name, h_pre, y_h_name, hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            y_h_list.append(y_h_name)
            h_pre = y_h_name
        # Expand RNN block (reverse)
        h_pre = initial_h_list[1]
        for x in reversed(x_list):
            y_h_name = x + '_rout_y_h'
            x_id_name = x + '_identity'
            identity_node = onnx.helper.make_node("Identity", [x], [x_id_name], name=x_id_name)
            new_nodes.append(identity_node)
            generated_nodes = make_RNN_block(x_id_name, reverse_w_name, reverse_r_name, reverse_b_name, h_pre, y_h_name,
                hidden_size, batch_size)
            new_nodes.extend(generated_nodes)
            reversed_y_h_list.append(y_h_name)
            h_pre = y_h_name
        reversed_y_h_list.reverse()
        # Prepare the output y
        final_y_h_list = []
        for i in range(seq_length):
            final_y_h_name = x_list[i] + '_concat_y_h'
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = [y_h_list[i], reversed_y_h_list[i]],
                outputs = [final_y_h_name],
                name = final_y_h_name,
                axis = 0
            )
            new_nodes.append(concat_node)
            final_y_h_list.append(final_y_h_name)
        if seq_length != 1:
            y_h_unsqueezed_output_list = []
            # Concat the outputs
            for y_h in final_y_h_list:
                unsqueezed_name = y_h + "_unsqueeze"
                unsqueeze_node = onnx.helper.make_node(
                    op_type = 'Unsqueeze',
                    inputs = [y_h],
                    outputs = [unsqueezed_name],
                    name = unsqueezed_name,
                    axes = [0]
                )
                new_nodes.append(unsqueeze_node)
                y_h_unsqueezed_output_list.append(unsqueezed_name)
            concat_node = onnx.helper.make_node(
                op_type = 'Concat',
                inputs = y_h_unsqueezed_output_list,
                outputs = [node.output[0]],
                name = node.name + '_final_concat',
                axis = 0
            )
            new_nodes.append(concat_node)
        else:
            # For single sequance, no need for concat.
            unsqueeze_node = onnx.helper.make_node(
                op_type = 'Unsqueeze',
                inputs = [final_y_h_list[0]],
                outputs = [node.output[0]],
                name = node.name + "_final_unsqueeze",
                axes = [0]
            )
            new_nodes.append(unsqueeze_node)
        # Connect the final y_h if needed
        if len(node.output) > 1 and node.output[1] != '':
            identity_node = onnx.helper.make_node(
                op_type = 'Identity',
                inputs = [final_y_h_list[-1]],
                outputs = [node.output[1]],
                name = node.name + '_Y_h_identity'
            )
            new_nodes.append(identity_node)
    else:
        helper.logger.error(f"Cannot expand RNN node {node.name}: invalid direction {direction}.")
        exit(1)
    g.node.extend(new_nodes)


def make_RNN_block(x_t, w_i, r_i, b_name, h_pre, y_h_name, hidden_size, batch_size):
    new_nodes = []
    # Squeeze x. The output shape should be [batch, input]
    x_t_squeezed = x_t + '_squeezed'
    squeeze_node = onnx.helper.make_node(
        op_type = "Squeeze",
        inputs = [x_t],
        outputs = [x_t_squeezed],
        name = x_t_squeezed,
        axes = [0]
    )
    new_nodes.append(squeeze_node)
    # Make Gemm for Xt*(W^T) + Wb + Rb. [batch, hidden]
    gemm_0_name = x_t + '_gemm_0'
    inputs = [x_t_squeezed, w_i]
    if b_name != '':
        inputs.append(b_name)
    gemm_0_node = onnx.helper.make_node(
            op_type = 'Gemm',
            inputs = inputs,
            outputs = [gemm_0_name],
            name = gemm_0_name,
            transB = 1
    )
    new_nodes.append(gemm_0_node)
    # Make Gemm for Ht-1*(R^T) [batch, hidden]
    h_prev_squeezed = x_t + '_ht-1'
    if h_pre == '':
        # No previous H found, make 0 constant node.
        h_prev_np = np.zeros((batch_size, hidden_size), dtype='float32')
        constant_node = helper.numpy_to_constant(h_prev_squeezed, h_prev_np)
        new_nodes.append(constant_node)
    else:
        # Squeeze h
        squeeze_node = onnx.helper.make_node(
            op_type = "Squeeze",
            inputs = [h_pre],
            outputs = [h_prev_squeezed],
            name = h_prev_squeezed,
            axes = [0]
        )
        new_nodes.append(squeeze_node)
    gemm_1_name = x_t + '_gemm_1'
    gemm_1_node = onnx.helper.make_node(
        op_type = 'Gemm',
        inputs = [h_prev_squeezed, r_i],
        outputs = [gemm_1_name],
        name = gemm_1_name,
        transB = 1
    )
    new_nodes.append(gemm_1_node)
    # Prepare ht(yh) [batch, hidden]
    ht_add_name = x_t + '_ht_add'
    ht_add_node = onnx.helper.make_node(
        op_type = "Add",
        inputs = [gemm_0_name, gemm_1_name],
        outputs = [ht_add_name],
        name = ht_add_name
    )
    new_nodes.append(ht_add_node)
    ht_f_name = x_t + '_ht_f'
    ht_f_node = onnx.helper.make_node(
        op_type = "Tanh",
        inputs = [ht_add_name],
        outputs = [ht_f_name],
        name = ht_f_name
    )
    new_nodes.append(ht_f_node)
    y_h_unsqueeze_node = onnx.helper.make_node(
        op_type = 'Unsqueeze',
        inputs = [ht_f_name],
        outputs = [y_h_name],
        name = y_h_name,
        axes = [0]
    )
    new_nodes.append(y_h_unsqueeze_node)
    return new_nodes
