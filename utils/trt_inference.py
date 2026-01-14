"""Generic TensorRT engine builder and inference runner."""

import os
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 — initializes CUDA context

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 input_shape: tuple = None) -> trt.ICudaEngine:
    """Build a TensorRT engine from an ONNX model file.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Path to save/load the TensorRT engine.
        fp16: Use FP16 precision if available.
        input_shape: Explicit input shape (e.g. (1, 3, 640, 640)) for models
                     with dynamic dimensions. If None, uses static shapes from ONNX.
    """
    if os.path.exists(engine_path):
        return load_engine(engine_path)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    # Verify network has outputs
    if network.num_outputs == 0:
        # Some ONNX models have unmarked outputs; mark all unconnected layers
        print(f"WARNING: No outputs in parsed network, attempting to mark outputs...")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                # Check if this tensor is not consumed by another layer
                is_leaf = True
                for k in range(network.num_layers):
                    other = network.get_layer(k)
                    for m in range(other.num_inputs):
                        if other.get_input(m) == out:
                            is_leaf = False
                            break
                    if not is_leaf:
                        break
                if is_leaf and out is not None:
                    network.mark_output(out)
        if network.num_outputs == 0:
            raise RuntimeError(f"Could not identify outputs in {onnx_path}")
        print(f"  Marked {network.num_outputs} outputs")

    print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        has_dynamic = any(s == -1 for s in shape)

        if has_dynamic:
            if input_shape is not None:
                resolved = list(input_shape)
            else:
                raise RuntimeError(
                    f"Input '{inp.name}' has dynamic dims {shape} but no "
                    f"input_shape was provided. Pass input_shape to build_engine()."
                )
            print(f"  Input '{inp.name}': {shape} -> {resolved}")
            profile.set_shape(inp.name, resolved, resolved, resolved)
        else:
            profile.set_shape(inp.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """Load a serialized TensorRT engine from disk."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class TRTInference:
    """Manages a TensorRT engine for inference with pre-allocated buffers."""

    def __init__(self, onnx_path: str, engine_path: str, fp16: bool = True,
                 input_shape: tuple = None):
        self.engine = build_engine(onnx_path, engine_path, fp16=fp16,
                                   input_shape=input_shape)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            host_mem = np.empty(shape, dtype=dtype)

            binding = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "device": device_mem,
                "host": host_mem,
            }

            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            self.bindings.append(int(device_mem))

    def infer(self, *input_arrays: np.ndarray) -> list[np.ndarray]:
        """Run inference with the given input arrays. Returns list of output arrays."""
        for i, arr in enumerate(input_arrays):
            np.copyto(self.inputs[i]["host"], arr.reshape(self.inputs[i]["shape"]))
            cuda.memcpy_htod_async(
                self.inputs[i]["device"],
                self.inputs[i]["host"],
                self.stream,
            )

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                binding = next(b for b in self.inputs if b["name"] == name)
            else:
                binding = next(b for b in self.outputs if b["name"] == name)
            self.context.set_tensor_address(name, int(binding["device"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        for out in self.outputs:
            results.append(out["host"].copy())
        return results

    def __del__(self):
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine
