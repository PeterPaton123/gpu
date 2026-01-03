"""
A simple linear regression model using ONNX, following the tutorial from the ONNX documentation (https://onnx.ai/onnx/intro/python.html#).
"""
import os

import numpy as np
import onnx
from onnx import load, numpy_helper, TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info
)


## Define input variables (e.g. inputs/outputs), specifying variable name,type and shape
# Shapes and types do not need to be specified, they can be inferred from the inputs and operations at runtime, however knowing shapes ahead of time can make inference faster, and more efficient, as intermediate tensors of the same shape can use the same memory and be computed in-place.
# Of course, ONNX cannot infer types and output shapes for custom operators outside the official operator set.

X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])

## Initializers
# Values such as model weights and model coefficients are not technically model inputs but constants. These can be specified as constants in the graph. 

A = numpy_helper.from_array(
    np.array([0.5, -0.6], dtype=np.float32), 
    name='A'
)
B = numpy_helper.from_array(
    np.array([0.4], dtype=np.float32), 
    name='B'
)

## Outputs
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

## Nodes are defined by an operations (an operator type), its inputs and outputs 
# For example, a matrix multiplication node that takes in nodes "X" and "A" and outputs a node "AX":

# Attributes
# Some operators need attributes to be specified, such as the padding for a convolution operation or transpose axes.
node_transpose = make_node("Transpose", ["A"], ["A_T"], perm=[1, 0])
node_1 = make_node("MatMul", ["X", "A_T"], ["XA"])
node_2 = make_node("Add", ["XA", "B"], ["Y"])

## A graph is built from a list of nodes, list of inputs, outputs and a graph name
graph = make_graph(
    [node_transpose, node_1, node_2], 
    "Linear Regression",
    [X], # Inputs 
    [Y], # Outputs
    [A, B] # Initializers
)

## Make the model
# (Optional): Specify graph metadata (https://onnx.ai/onnx/intro/concepts.html#metadata)
graph.doc_string = "A simple linear regression model: Y = X * A + B"
onnx_model = make_model(graph, producer_name="example_linear_regression", model_version=1)

## Check the model is consistent (see: https://onnx.ai/onnx/intro/python.html#checker-and-shape-inference)
print("Verifying the model...")
try:
    onnx.checker.check_model(onnx_model)
    print("\tThe model is consistent!")
except onnx.checker.ValidationError as e:
    print(f"\tThe model is not consistent: {e}")

## Serialization
# For a thorough explanation of the serialization process, see: https://onnx.ai/onnx/intro/python.html#data-serialization
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

file_size: int = os.path.getsize("linear_regression.onnx")
print(f"Serialized model file size: {file_size} Bytes")

## Deserialization: The graph can be restored by deserializing the model protobuf
# Any model can be serialized this way unless they are bigger than 2 GB (source: https://onnx.ai/onnx/intro/python.html#model-serialization)
with open("linear_regression.onnx", "rb") as f:
    loaded_onnx_model = load(f)

if not (loaded_onnx_model == onnx_model):
    print("The deserialized model is not equal to the original model")
    exit(1)

## Display the model
# print(onnx_model)
print("Initializers:")
for init in onnx_model.graph.initializer:
    print(f"\tName: {init.name}, Shape: {init.dims}, Data Type: {init.data_type}")
