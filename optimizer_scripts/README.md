# Converter Scripts

[![pipeline status](http://192.168.200.1:8088/jiyuan/converter_scripts/badges/master/pipeline.svg)](http://192.168.200.1:8088/jiyuan/converter_scripts/commits/master)

This project collects various optimization scripts and converter scritps for
Kneron toolchain. This collection does not include the Keras to ONNX converter
and the Caffe to ONNX converter. They are in seperate projects.

**The scripts not listed below are used as libraries and cannot be used
directly.**

## onnx2onnx.py

### 1.1. Description

General optimizations on ONNX model for Kneron toolchain. Though Kneron
toolchains are designed to take ONNX models as input, they have some
restrictions on the models (e.g. inferenced shapes for all value_info). Thus, we
have this tool to do some general optimization and conversion on ONNX models.
**Notice that this script should take an valid ONNX model as input.** It cannot
turn an invalid ONNX model into a valid one.

### 1.2. Basic Usage

```bash
python onnx2onnx.py input.onnx -o output.onnx
```

### 1.3. Optimizations Included

* Fusing BN into Conv.
* Fusing BN into Gemm.
* Fusing consecutive Gemm.
* Eliminating Identify layers and Dropout layers.
* Eliminating last shape changing nodes.
* Replacing initializers into Constant nodes.
* Replacing global AveragePool with GAP.
* Replacing Squeeze and Unsqueeze with Reshape.
* Replacing 1x1 depthwise with BN.
* Inferencing Upsample shapes.
* Transposing B in Gemm.

## pytorch2onnx.py

### 2.1. Description

Convert Pytorch models or Pytorch generated ONNX models into Kneron toolchain
compatible ONNX files. This script include most of the optimizations in
`onnx2onnx.py`. It also includes some optimizations for Pytorch model only.

### 2.2. Basic Usage

```bash
# Take Pytorch model name, input channel number, input height, input width
python pytorch2onnx.py input.pth output.onnx --input-size 3 224 224
# Or take Pytorch exported ONNX.
python pytorch2onnx.py input.onnx output.onnx
```

### 2.3. Optimizations Included

* Adding name to nodes.
* Unsqueeze nodes constant folding.
* Reshape nodes constant folding.
* Optimizations in `onnx2onnx.py`.

## editor.py

### 3.1. Description

This is an simple ONNX editor which achieves the following functions:

* Add nop BN or Conv nodes.
* Delete specific nodes or inputs.
* Cut the graph from certain node (Delete all the nodes following the node).
* Reshape inputs and outputs

### 3.2 Usage

```
usage: editor.py [-h] [-c CUT_NODE [CUT_NODE ...]]
                 [--cut-type CUT_TYPE [CUT_TYPE ...]]
                 [-d DELETE_NODE [DELETE_NODE ...]]
                 [--delete-input DELETE_INPUT [DELETE_INPUT ...]]
                 [-i INPUT_CHANGE [INPUT_CHANGE ...]]
                 [-o OUTPUT_CHANGE [OUTPUT_CHANGE ...]]
                 [--add-conv ADD_CONV [ADD_CONV ...]]
                 [--add-bn ADD_BN [ADD_BN ...]]
                 in_file out_file

Edit an ONNX model. The processing sequense is 'delete nodes/values' -> 'add
nodes' -> 'change shapes'. Cutting cannot be done with other operations
together

positional arguments:
  in_file               input ONNX FILE
  out_file              ouput ONNX FILE

optional arguments:
  -h, --help            show this help message and exit
  -c CUT_NODE [CUT_NODE ...], --cut CUT_NODE [CUT_NODE ...]
                        remove nodes from the given nodes(inclusive)
  --cut-type CUT_TYPE [CUT_TYPE ...]
                        remove nodes by type from the given nodes(inclusive)
  -d DELETE_NODE [DELETE_NODE ...], --delete DELETE_NODE [DELETE_NODE ...]
                        delete nodes by names and only those nodes
  --delete-input DELETE_INPUT [DELETE_INPUT ...]
                        delete inputs by names
  -i INPUT_CHANGE [INPUT_CHANGE ...], --input INPUT_CHANGE [INPUT_CHANGE ...]
                        change input shape (e.g. -i 'input_0 1 3 224 224')
  -o OUTPUT_CHANGE [OUTPUT_CHANGE ...], --output OUTPUT_CHANGE [OUTPUT_CHANGE ...]
                        change output shape (e.g. -o 'input_0 1 3 224 224')
  --add-conv ADD_CONV [ADD_CONV ...]
                        add nop conv using specific input
  --add-bn ADD_BN [ADD_BN ...]
                        add nop bn using specific input
```

### 3.3. Example

Here is an example of when and how to use the editor.py.

```bash
# In the `res` folder, there is a vdsr model from tensorflow.
# We need to convert this model firstly.
./tf2onnx.sh res/vdsr_41_20layer_1.pb res/tmp.onnx images:0 output:0
# This onnx file seems valid. But, it's channel last for the input and output.
# It is using Traspose to convert to channel first, affacting the performance.
# Thus, here we use the editor to delete these Transpose and reset the shapes.
python editor.py debug.onnx new.onnx -d Conv2D__6 Conv2D_19__84 -i 'images:0 1 3 41 41' -o 'output:0 1 3 41 41'
# Now, it has no Transpose and take channel first inputs directly.
```

## test_models_opt.py

### 4.1. Description
Compare all original and optimized onnx models under a specified directory.
Using different endings to locate original and optimized model paths. Apply 
onnxruntime inference to the models, and compare the results from original 
and optimized models. Calculate basic statistics and store to a csv file.

### 4.2. Usage

```bash
python DIR ending1 ending2 csv_out_file -p=Y/N

# csv_out_file is file path for the stats data.
# -p --plot is the plot option, if Y, stats plots will be generated.
```

### 4.3. Statistics
* max_rel_diff
* max_abs_diff
* mean_rel_diff
* mean_abs_diff
* std_rel_diff
* std_abs_diff
* acc_with_diff_precision
* percentile

### 4.4. Plots
* Max Relative Difference Histogram
* Max Absolute Difference Histogram
* Rel_diff Percentiles of Raw and Optimized Models
* Abs_diff Percentiles of Raw and Optimized Models
* Accuracies with Different Precisions

## tensorflow2onnx.py

### 5.1. Description
Convert and optimize tensorflow models. If input file is frozen tensorflow .pb model,
convert to onnx model and do the custmized optimization afterwards. If input model is already
onnx model, apply optimization and save optimized model.

### 5.2 Dependency

This scripts depends on the tensorflow-onnx project. Please [check and install it](https://github.com/onnx/tensorflow-onnx/tree/r1.5) before using this script. We currently support up to version 1.5.5. For other versions, you may need to try it our yourself.

### 5.3. Basic Usage
```bash
python tensorflow2onnx.py in_file out_file -t=True/False

# -t --test, is the option for test mode, if True, shape change after input will not be eliminated.
```

### 5.4. Model Save Paths
`in_file` is the input model path, `out_file` specifies output optimized model path.
If input file is `.pb` model, an unoptimized onnx model will be saved to the output directory as well.

