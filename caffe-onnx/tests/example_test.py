import os.path
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
from caffe import layers as L, params as P
import sys
sys.path.insert(0, "./onnx-caffe/")
import frontend
from onnx_tf.backend import prepare
import cntk as C
 
model_caffe = 'lenet.prototxt'
weights_caffe = 'lenet.caffemodel'
net = caffe.Net(model_caffe, weights_caffe, caffe.TEST)

img = np.random.rand(1, 1, 28, 28).astype('float32')
net.blobs['data'].data[...] = img
net.forward()
ans_caffe = net.forward()[net._layer_names[-1]][0]

converter = frontend.CaffeFrontend()
converter.loadFromFile(model_caffe, weights_caffe)
onnx_model = converter.convertToOnnx()
converter.saveToFile('lenet.onnx')
tf_rep = prepare(onnx_model)
ans_onnx_tf = tf_rep.run(img)

z = C.Function.load('lenet.onnx', device=C.device.cpu(), format=C.ModelFormat.ONNX)
ans_onnx_cntk = z.eval(img)

preds_caffe = ans_caffe
preds_onnx_cntk = ans_onnx_cntk
preds_onnx_tf = ans_onnx_tf

threshold = 0.01
abs_result = np.absolute(preds_caffe - preds_onnx_cntk)
if (abs_result > threshold).any():
	print("Wrong values (threshold: {}):".format(threshold))
	wv = abs_result[np.where(abs_result > threshold)]
	print(sorted(wv, reverse=True)[:10])
	print("Count: {} - {:.2f}%".format(len(wv), len(wv) / abs_result.size * 100))
	print("Difference:")
	#print(abs_result)
else:
	print("CNTK Test passed (threshold: {})".format(threshold))

abs_result = np.absolute(preds_caffe - preds_onnx_tf)
print(abs_result.shape)
if (abs_result > threshold).any():
	print("Wrong values (threshold: {}):".format(threshold))
	wv = abs_result[np.where(abs_result > threshold)]
	print(sorted(wv, reverse=True)[:10])
	print("Count: {} - {:.2f}%".format(len(wv), len(wv) / abs_result.size * 100))
	print("Difference:")
	#print(abs_result)
else:
	print("TF Test passed (threshold: {})".format(threshold))
