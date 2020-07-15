import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file')
    parser.add_argument('-save_path', metavar='saved model path', help='an output folder path')
    args = parser.parse_args()

    model_path = os.path.abspath(args.tflite)
    model_save_path = os.path.abspath(args.save_path)

    # change working directory
    os.chdir(os.path.abspath('./onnx_tflite/'))
    os.system("python ./tflite2onnx.py -tflite " + model_path + " -save_path " + model_save_path)

    print('model_path: ' + model_path)
    print('model_save_path: ' + model_save_path)
    print('done!')
