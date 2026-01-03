# Open Neural Network Exchange (ONNX)

Today there are many machine learning frameworks (scikit-learn, Tensorflow, PyTorch, JAX, HuggingFace, etc) which each implement inference in different ways. Deploying models requires replicating the model training environments, most of the time via docker. ONNX ([onnx.ai](https://onnx.ai/), writted by the PyTorch team at Meta) attempts to unify the serialisation and deployment of trained models for deployment in a framework-agnostic way. Once a model is serialised using ONNX, the inference environment needs to only support ONNXRuntime and the serialised model protobuf to execute the computation graph defined in ONNX. 

ONNX implements a common set of [operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) used in machine learning models (e.g. matrix multiplication, attention, batch normalisation, etc) and a common protobuf file format for serialising models. See [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html) for a basic introduction. 

`linear_regression.py`: An example of a basic linear regression model, the inference graph of operations is defined, serialised and written to protobuf. Then it is de-serialised and used in inference. This file largely follows the [Python example](https://onnx.ai/onnx/intro/python.html) in the documentation.

`neural_network.py`: ONNX supports user-defined operators, which belong to user-defined 'domains'. This file defines a new operation corresponding to a single layer of a multi-layer perception dense neural network. By invoking this operation multiple times we can build the inference graph for a neural network with arbitrary layers.

`sklearn_converter.py`: ONNX supports intergrations with prominent machine learning frameworks. Once a model is trained it can be simply serialised into ONNX to simplify the deployment and inference environment. Essentially a converter defines the mapping from operations in a specific machine learning framework to ONNX's framework-independent primatives. In this example, we train a sklearn classifier on the Iris dataset. The model is converted to ONNX using [skl2onnx](https://github.com/onnx/sklearn-onnx) and inference is performed solely in ONNX. See the documentation for a full list of [supported converters](https://onnx.ai/onnx/intro/converters.html).


