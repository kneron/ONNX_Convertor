from keras.layers import *
from keras.models import *
import numpy as np
import sys

def convert(model):
    """
    :type model: Model
    :param model:
    :return:
    """
    shape = model.input_shape
    inp = Input(shape[1:])
    rgba2yynnconv = Conv2D(4,3,padding='same',use_bias=False)

    x = rgba2yynnconv(inp)
    x = model(x)
    weights = rgba2yynnconv.get_weights()[0]
    weights = np.zeros_like(weights)
    weights[1,1,:3,:2] = np.array([[[[0.299],
                                          [0.587],
                                          [0.114]]]])
    weights[1,1,3,2:] = 1.
    rgba2yynnconv.set_weights([weights])
    return Model(inp, x)

if len(sys.argv) < 3:
    print("Need 2 arguments.\npython {} input.hdf5 output.hdf5", sys.argv[0])
    exit(1)
model = load_model(sys.argv[1])
new_model = convert(model)
save_model(new_model, sys.argv[2])
