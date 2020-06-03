# Caffe Frontend for Onnx

## Target

Read an HDF5 format Caffe model into Onnx.

## Installation

### Prerequisites

* **Python 3**. Python 3.5 is recommanded. You can use Anaconda to create the virtual environment.
  For example, run `conda create -n kmc python=3.5` and then run `conda activate kmc`.
* **onnx**. Recommand installing with `pip install onnx`.
* **caffe**. Recommand using our docker image using `docker pull mrwhoami/kmc_docker`
* **tensorflow**. Recommand inhstalling with `conda install -c conda-forge tensorflow`.

Note:

* The environment is tested under both Ubuntu and Windows.
* To run the test code under the test folder, you may need additional packages(optional).

### Optional packages (needed by test code)

* jupyter
* cntk

## Basic Usage

1. Go under the `onnx-caffe` folder which holds the `generate_onnx.py`.
2. Try `python generated_onnx.py -h` to check the environment.
3. Run `python generated_onnx.py -o outputfile.onnx -n inputfile.prototxt -w inputfile.caffemodel` to convert the model,
   where `-o` specify the output file name and the last parameter is the input file name.

You can check the ipython notebooks in the test folder for more details.

## Advanced Usage

In your Python program, you can import frontend and make full use of our module.

First of all, you can load your caffe model both by file or the model object.

```python
import frontend
converter = frontend.CaffeFrontend()
converter.loadFromFile(YOUR_PATH_HERE) # Load from the file
converter.loadFromModel(YOUR_MODEL_OBJECT) # Load from the model object
```

Then, you can do the conversion using `convertToOnnx` function. This function will return the converted Onnx model object. If you forget to get the return value, you can also get the converted model object later by the `omodel` class variable.

```python
onnx_obj1 = converter.convertToOnnx()
onnx_onj2 = converter.omodel
```

After that, you can play with the Onnx object as your with. And you can save the model to a file as well.

```python
converter.saveToFile(YOUR_PATH_HERE)
```

## Developer

To contribute, please read the [Developer Guide](DevGuide.md) first.
