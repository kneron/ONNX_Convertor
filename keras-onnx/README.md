# Keras Frontend for Onnx

## Target

Read an HDF5 format Keras model into Onnx.

## Installation

### Prerequisites

* **Python 3**. Python 3.5 is recommanded. You can use Anaconda to create the virtual environment.
  For example, run `conda create -n kmc python=3.5` and then run `conda activate kmc`.
* **onnx**. Recommand installing with `pip install onnx==1.4.1`.
* **keras**. Recommand installing with `pip install keras==2.2.4`.
* **tensorflow**. Recommand inhstalling with `conda install -c conda-forge tensorflow`.

Note:

* The environment is tested under both Ubuntu and Windows.
* To run the test code under the test folder, you may need additional packages(optional).

### Optional packages (needed by test code)

* jupyter
* cntk

## Basic Usage

1. Go under the `onnx-keras` folder which holds the `generate_onnx.py`.
2. Try `python generated_onnx.py -h` to check the environment.
3. Run `python generated_onnx.py -o outputfile.onnx inputfile.hdf5` to convert the model,
   where `-o` specify the output file name and the last parameter is the input file name.

Other command line options:

* `-D`: Enable debug output.
* `-C`: Enable compatibility mode.
* `-O [level]`: Enable optimizations at a specific level. `-O 1` includes eliminating dropout layers. `-O 2` includes fusing paddings into the next layers and replacing some average pooling layers with global average poolibng. `-O 3` includes fusing batch normalization into convolution layers. All the optimization with greater number also includes the optimizations in the lower levels, for example, `-O 3` also includes all the optimizations in `-O 2` and `-O 1`.
* `-c config.json`: Take a json file to classify the Lambda layers by their names. This helps the backend for further computation. [Here](custom.json) is an example.

## Advanced Usage

In your Python program, you can import frontend and make full use of our module.

First of all, you can load your keras model both by file or the model object.

```python
import frontend
converter = frontend.KerasFrontend()
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

## Supported Operators

Please check [supported operators](Operators.md).

## Developer

To contribute, please read the [Developer Guide](DevGuide.md) first.
