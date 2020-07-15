import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert a tflite model into an onnx file.')
    parser.add_argument('-tflite', metavar='tflite model path', help='an input tflite file')
    parser.add_argument('-save_path', metavar='saved model path', help='an output folder path')
    parser.add_argument('-release_mode', metavar='is release mode', help='True if no traspose front end needed')
    args = parser.parse_args()

    model_path = os.path.abspath(args.tflite)
    model_save_path = os.path.abspath(args.save_path)
    is_release_mode = True if args.release_mode == 'True' else False

    # change working directory
    os.chdir(os.path.abspath('./onnx_tflite/'))
    os.system("python ./tflite2onnx.py -tflite " + model_path + " -save_path " + model_save_path + " -release_mode " + str(is_release_mode))

    print('is_release_mode: ' + str(is_release_mode))
    print('model_path: ' + model_path)
    print('model_save_path: ' + model_save_path)
    print('done!')
