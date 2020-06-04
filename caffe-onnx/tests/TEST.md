
#### Tested models

The models are tested by comparing the inference given by the original caffe model and the onnx model with different backend. Here we list the tested models in the table, in which  o indicate that there are differences between the original inference and the answer given by converted model. However, all the layers have passed the unit test.

Models | CNTK | Tensorflow
:-----:|:-----:|:----------:|
[LeNet]| √ | √ |
[Inception V2]| √ | o |
[Inception V3]| √ | o |
[Inception V4]| o | o |
[Inception ResNet V2]| √ | √ |
[DenseNet]|   √   |   √   |
[ResNet V1]|   √   |   √   |
[ResNet V2]|   √   |   √   |
[MobileNet V1]|   √   |   o   |
[MobileNet V2]|   √   |   o   |

#### Problems
However, there are still some flaws in the CNTK. So there might be some difference between the inferences. Here we list the problems.
1.	CNTK cannot deal with epsilon other than 1e-5 in the BatchNorm layer.
2. In caffe, the output size of pooling layer is calculated by formular
		*ceil((bottom_size + 2\*pad - kernel_size) / stride) + 1*
	while in other models it is calculated by *floor*. To fix this problem, pad is adjusted elaborately and count_include_pad parameter is assigned to zero. However, CNTK cannot deal with this parameter and will count average with zero in pad.
