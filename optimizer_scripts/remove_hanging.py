import sys
from tools import other, replacing
import onnx
from onnx import optimizer

m = onnx.load(sys.argv[1])
replacing.replace_initializer_with_Constant(m.graph)
other.topological_sort(m.graph)
m = optimizer.optimize(m, ['eliminate_deadend'])

onnx.save(m, sys.argv[2])