# Model with Loop Node (LSTM) Processing Manual

Currently, many models exported by tf2onxx have the Loop nodes and may contains very complicated structures.

On how to use `tf2onnx` to export onnx, please check its Github repository <https://github.com/onnx/tensorflow-onnx>. Note that the onnx opset should be version 11.

## Cut Loop Nodes

1. Use [Netron](https://netron.app/) to check the model structure. The cut points should be before the Loop nodes. We recommend cut after Concat nodes if any.
2. Use script `force_node_cut.py` to cut the onnx model. The command shall be like `python ONNX_Convertor/optimizer_scripts/force_node_cut.py input.onnx output.onnx -c value_name -s "1 3"`. In the command above, you may replace `input.onnx` and `output.onnx` with the actual file names. `value_name` is the expected final node's output name. And you shall replace `"1 3"` with tha actuall expected output name. This shape is for reference only. It will only be applied to the output model if the onnx library cannot inference the output shape. In fact, you can cut the graph from multiple points at the same time, for example: `python ONNX_Convertor/optimizer_scripts/force_node_cut.py input.onnx output.onnx -c value_1_name value_2_name -s "1 3" "1 255"`.
3. Use `tensorflow2onnx.py` to check the output model shape and optimize the model. If error raised during this process, please go back to step 1 and adjust the output shape according to the error message.

## Final Shape Adjustment

1. Use `onnx2onnx.py` to automatically optimize the model.
2. Eliminate remaining Reshape nodes using `force_shape_adjust.py`.
3. Use `tensorflow2onnx.py` to check the output model shape and optimize the model again.
4. Change the batch size if the batch size is not 1, adjust it using `batch_size_change.py`.
