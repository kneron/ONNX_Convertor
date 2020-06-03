import onnx
import sys
import json

from tools import special

if len(sys.argv) != 3:
    print("python norm_on_scaled_onnx.py input.onnx input.json")
    exit(1)

# Modify onnx
m = onnx.load(sys.argv[1])
special.add_0_5_to_normalized_input(m)
onnx.save(m, sys.argv[1][:-4] + 'norm.onnx')

# Change input node
origin_file = open(sys.argv[2], 'r')
origin_json = json.load(origin_file)
origin_json["input_node"]["output_datapath_radix"] = [8]
new_json_str = json.dumps(origin_json)

# Modify json
file = open(sys.argv[1][:-4] + 'norm.onnx' + '.json', 'w')
s = """{{
	\"{0}\" :
	{{
		\"bias_bitwidth\" : 16,
		\"{0}_bias\" : [15],
		\"{0}_weight\" : [3,3,3],
		\"conv_coarse_shift\" : [-4,-4,-4],
		\"conv_fine_shift\" : [0,0,0],
		\"conv_total_shift\" : [-4,-4,-4],
		\"cpu_mode\" : false,
		\"delta_input_bitwidth\" : [0],
		\"delta_output_bitwidth\" : 8,
		\"flag_radix_bias_eq_output\" : true,
		\"input_scale\" : [[1.0,1.0,1.0]],
		\"output_scale\" : [1.0, 1.0, 1.0],
		\"psum_bitwidth\" : 16,
		\"weight_bitwidth\" : 8,
		\"input_datapath_bitwidth\" : [8],
		\"input_datapath_radix\" : [8],
		\"working_input_bitwidth\" : 8,
		\"working_input_radix\" : [8],
		\"working_output_bitwidth\" : 16,
		\"working_output_radix\" : 15,
		\"output_datapath_bitwidth\" : 8,
		\"output_datapath_radix\" : 7
	}},\n""".format('input_norm')
file.write(s + new_json_str[1:])
file.close()
origin_file.close()
