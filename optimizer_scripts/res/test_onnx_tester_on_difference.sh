#!/bin/bash

python onnx_tester.py /test_models/mobilenet_v2_224.onnx /test_models/mobilenet_v2_224.cut.onnx
if [ $? -eq 0 ]; then
  echo "Those two model results should be different!"
  exit 1
fi

exit 0
