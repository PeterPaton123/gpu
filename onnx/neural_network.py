"""
A simple 2-layer neural network using ONNX custom functions.
ONNX allows frameworks to define their own custom nodes, these can be used multiple times in a graph.
E.g. Here we define a custom node for a single dense layer of a neural network.
"""
import os

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, load
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
    make_function,
)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

def make_linear_relu_function(domain: str, opset_imports: list) -> onnx.FunctionProto:
    """
    Create an ONNX function that performs: Y = ReLU(X @ W + b)
    
    This is equivalent to a single dense/linear layer with ReLU activation,
    commonly used as a building block in neural networks.
    
    Args:
        domain: Custom domain name for the function (e.g., 'custom.nn')
        opset_imports: List of opset imports for standard ONNX ops
    
    Returns:
        FunctionProto: Reusable ONNX function definition
    
    Function signature:
        Inputs:  X (data), W (weights), b (bias)
        Outputs: Y (activated output)
    """
    matmul_node = make_node('MatMul', ['X', 'W'], ['XW'])
    add_node = make_node('Add', ['XW', 'b'], ['Z'])
    relu_node = make_node('Relu', ['Z'], ['Y'])
    
    return make_function(
        domain,                     # Domain name for this custom op
        'LinearRelu',               # Function name (op type when used)
        ['X', 'W', 'b'],            # Input parameter names
        ['Y'],                      # Output parameter names
        [matmul_node, add_node, relu_node],  # Nodes in topological order
        opset_imports,              # Required opsets for internal ops
        []                          # Attributes (none for this function)
    )

# Define custom domain for our neural network functions
CUSTOM_DOMAIN = 'custom.nn'
# Opset imports: standard ONNX ops (version 14) + our custom domain (version 1)
opset_imports = [
    make_opsetid("", 14),           # Standard ONNX operators
    make_opsetid(CUSTOM_DOMAIN, 1)  # Our custom LinearRelu function
]
# Create the reusable LinearRelu function
linear_relu_fn = make_linear_relu_function(CUSTOM_DOMAIN, opset_imports)

IN_FEATURES = 4      # Input dimension
HIDDEN_DIM = 8       # Hidden layer dimension  
OUT_FEATURES = 2     # Output dimension

## Graph Inputs
# X is the only runtime input; weights are stored as initializers
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, IN_FEATURES])

## Initializers (model parameters - would be learned during training)
W1 = numpy_helper.from_array(
    np.random.randn(IN_FEATURES, HIDDEN_DIM).astype(np.float32) * 0.1,
    name='W1'
)
b1 = numpy_helper.from_array(
    np.zeros(HIDDEN_DIM, dtype=np.float32),
    name='b1'
)
W2 = numpy_helper.from_array(
    np.random.randn(HIDDEN_DIM, OUT_FEATURES).astype(np.float32) * 0.1,
    name='W2'
)
b2 = numpy_helper.from_array(
    np.zeros(OUT_FEATURES, dtype=np.float32),
    name='b2'
)

## Graph Output
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, OUT_FEATURES])
layer1_node = make_node(
    'LinearRelu',
    ['X', 'W1', 'b1'],
    ['H1'],
    domain=CUSTOM_DOMAIN # Must specify custom domain
)
layer2_node = make_node(
    'LinearRelu',
    ['H1', 'W2', 'b2'],
    ['Y'],
    domain=CUSTOM_DOMAIN
)

# Assemble the graph
graph = make_graph(
    nodes=[layer1_node, layer2_node],   # Nodes in execution order
    name='TwoLayerMLP',
    inputs=[X],
    outputs=[Y],
    initializer=[W1, b1, W2, b2]
)

graph.doc_string = f"""
Two-layer Multi-Layer Perceptron (MLP)

Architecture:
    Input:  (batch_size, {IN_FEATURES})
    Layer1: LinearRelu({IN_FEATURES} -> {HIDDEN_DIM})
    Layer2: LinearRelu({HIDDEN_DIM} -> {OUT_FEATURES})
    Output: (batch_size, {OUT_FEATURES})

Each LinearRelu layer computes: Y = ReLU(X @ W + b)
"""

onnx_model = make_model(
    graph,
    opset_imports=opset_imports,
    functions=[linear_relu_fn],
    producer_name='neural_network_example',
    model_version=1
)

print("Validating model...")
try:
    check_model(onnx_model)
    print("\t✓ Model is valid!")
except onnx.checker.ValidationError as e:
    print(f"\t✗ Validation failed: {e}")
    exit(1)

# Serialization
MODEL_PATH = 'neural_network.onnx'
with open(MODEL_PATH, 'wb') as f:
    f.write(onnx_model.SerializeToString())
file_size = os.path.getsize(MODEL_PATH)
print(f"Saved model to '{MODEL_PATH}' ({file_size} bytes)")

print("\n--- Inference Test ---")

# Create test input: batch of 3 samples, each with IN_FEATURES dimensions
test_input = np.random.randn(3, IN_FEATURES).astype(np.float32)
session = ReferenceEvaluator(onnx_model)
outputs = session.run(None, {'X': test_input})

print(f"Input shape:  {test_input.shape}")
print(f"Output shape: {outputs[0].shape}")
print(f"\nInput:\n{test_input}")
print(f"\nOutput:\n{outputs[0]}")
