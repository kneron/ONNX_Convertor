# Caffe Converter for Onnx

## Target

Convert an Caffe model into Onnx.

## Installation

We recommand using docker `kneron/intel-caffe` to get the full workable environment and the converter itself.

### Prerequisites

* **Python 3**. Python 3.5 is recommanded. You can use Anaconda to create the virtual environment.
  For example, run `conda create -n kmc python=3.5` and then run `conda activate kmc`.
* **onnx**. Recommand installing with `pip install onnx`.
* **caffe**. [intel/caffe](https://github.com/intel/caffe) is recommended.

## Basic Usage

1. Go under the `onnx-caffe` folder which holds the `generate_onnx.py`.
2. Try `python generated_onnx.py -h` to check the environment.
3. Run `python generated_onnx.py -o outputfile.onnx -n inputfile.prototxt -w inputfile.caffemodel` to convert the model,
   where `-o` specify the output file name and the last parameter is the input file name.

## Developer

This project is very similar to keras-onnx. You can check the development guide in this project to get some useful information.
