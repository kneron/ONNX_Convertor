import onnx
from . import helper

def check_operator_type(g):
    onnx_operators = {
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "ArgMax",
        "ArgMin",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "AveragePool",
        "BatchNormalization",
        "BitShift",
        "Cast",
        "Ceil",
        "Clip",
        "Compress",
        "Concat",
        "ConcatFromSequence",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "ConvInteger",
        "ConvTranspose",
        "Cos",
        "Cosh",
        "CumSum",
        "DepthToSpace",
        "DequantizeLinear",
        "Det",
        "Div",
        "Dropout",
        "Einsum",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "EyeLike",
        "Flatten",
        "Floor",
        "GRU",
        "Gather",
        "GatherElements",
        "GatherND",
        "Gemm",
        "GlobalAveragePool",
        "GlobalLpPool",
        "GlobalMaxPool",
        "Greater",
        "HardSigmoid",
        "Hardmax",
        "Identity",
        "If",
        "InstanceNormalization",
        "IsInf",
        "IsNaN",
        "LRN",
        "LSTM",
        "LeakyRelu",
        "Less",
        "Log",
        "LogSoftmax",
        "Loop",
        "LpNormalization",
        "LpPool",
        "MatMul",
        "MatMulInteger",
        "Max",
        "MaxPool",
        "MaxRoiPool",
        "MaxUnpool",
        "Mean",
        "Min",
        "Mod",
        "Mul",
        "Multinomial",
        "Neg",
        "NonMaxSuppression",
        "NonZero",
        "Not",
        "OneHot",
        "Or",
        "PRelu",
        "Pad",
        "Pow",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
        "RNN",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "Range",
        "Reciprocal",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "Relu",
        "Reshape",
        "Resize",
        "ReverseSequence",
        "RoiAlign",
        "Round",
        "Scan",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Selu",
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "Shape",
        "Shrink",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Size",
        "Slice",
        "Softmax",
        "Softplus",
        "Softsign",
        "SpaceToDepth",
        "Split",
        "SplitToSequence",
        "Sqrt",
        "Squeeze",
        "StringNormalizer",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "TfIdfVectorizer",
        "ThresholdedRelu",
        "Tile",
        "TopK",
        "Transpose",
        "Unique",
        "Unsqueeze",
        "Upsample",
        "Where",
        "Xor",
        "Celu",
        "DynamicQuantizeLinear",
        "GreaterOrEqual",
        "LessOrEqual",
        "MeanVarianceNormalization",
        "NegativeLogLikelihoodLoss",
        "Range",
        "SoftmaxCrossEntropyLoss"
    }

    unsupported_operators = {
        "Abs",
        "Acos",
        "Acosh",
        "ArgMax",
        "ArgMin",
        "BitShift",
        "Compress",
        "ConcatFromSequence",
        "ConvInteger",
        "Cos",
        "Cosh",
        "CumSum",
        "Det",
        "EyeLike",
        "Greater",
        "If",
        "IsInf",
        "IsNaN",
        "Less",
        "Log",
        "LogSoftmax",
        "Loop",
        "LpPool",
        "MatMulInteger",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
        "RandomNormal",
        "RandomUniform",
        "RandomUniformLike",
        "ReduceL1",
        "ReduceL2",
        "ReduceProd",
        "ReverseSequence",
        "Round",
        "Scan",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "Sign",
        "Sin",
        "Sinh",
        "Size",
        "SplitToSequence",
        "StringNormalizer",
        "Tan",
        "TfIdfVectorizer",
        "TopK",
        "Unique",
        "Where",
        "Xor",
        "DynamicQuantizeLinear",
        "GreaterOrEqual",
        "LessOrEqual",
        "MeanVarianceNormalization",
        "NegativeLogLikelihoodLoss",
        "SoftmaxCrossEntropyLoss"
    }
    not_in_onnx_model_operators = set()
    not_supported_model_operators = set()
    for node in g.node:
        if node.op_type in unsupported_operators:
            not_supported_model_operators.add(node.op_type)
        elif node.op_type.startswith("Kneron"):
            pass
        elif node.op_type not in onnx_operators:
            not_in_onnx_model_operators.add(node.op_type)
    if len(not_supported_model_operators) != 0:
        helper.logger.warn(f"The following operatos are currently not well supported by our toolchain: {not_supported_model_operators}")
    if len(not_in_onnx_model_operators) != 0:
        helper.logger.error(f"The following operators are not in ai.onnx(standard onnx opset). Please modify your model to remove them: {not_in_onnx_model_operators}")
        exit(1)
