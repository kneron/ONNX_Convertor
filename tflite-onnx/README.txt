HowTo:

	python generate_onnx.py -tflite YOUR_TFLITE_PATH -save_path ONNX_SAVE_DIR 


Run Example:
	1. convert tflite to onnx 
            python generate_onnx.py -tflite ./example/example.tflite -save_path ./example -release_mode True
	2. convert onnx to kneron-onnx
	    python ../optimizer_scripts/onnx2onnx.py ./example/example.onnx
        

Tested Environment:
	
	tensorflow==1.15
	onnx==1.4.1


