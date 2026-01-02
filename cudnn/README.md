## cuDNN

cuDNN is a GPU-accelerated library that implements neural network primitives.

The library exposes a C API (`libcudnn`), with the core implementation written in C/C++. A C++ Frontend API is provided on top of the C API and can be used to access all backend functionality. Use of the frontend is recommended, as it provides a less verbose and more expressive interface.

A Python frontend interface is implemented on top of the C++ frontend using PyBind11.

cuDNN exposes a declarative programming model in its frontend APIs: users describe computation as a graph of tensor operations, and the library manages how these computations are executed on the GPU.
