'''Native TensorRT adapter for static, single-input YOLO engines.'''

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class TensorRTSession:
    def __init__(self, path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError('TensorRT Python bindings are unavailable') from exc

        self.cuda = ctypes.CDLL('libcudart.so')
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self.cuda.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_size_t, ctypes.c_int,
                                              ctypes.c_void_p]
        self.cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Could not load TensorRT engine: {path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context')

        inputs, outputs = [], []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dim < 1 for dim in shape):
                raise ValueError(f'Only static TensorRT shapes are supported: {shape}')
            if self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise ValueError(f'Only float32 TensorRT I/O is supported: {name}')
            target = inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
            target.append((name, shape))
        if len(inputs) != 1 or not outputs:
            raise ValueError('Expected one TensorRT input and at least one output')
        self.input_name, self.input_shape = inputs[0]
        self.input_device = ctypes.c_void_p()
        self.output_buffers = []
        self.stream = ctypes.c_void_p()
        try:
            self._check(self.cuda.cudaStreamCreate(ctypes.byref(self.stream)),
                        'cudaStreamCreate')
            self._check(self.cuda.cudaMalloc(
                ctypes.byref(self.input_device),
                int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
            ), 'cudaMalloc(input)')
            if not self.context.set_tensor_address(self.input_name, self.input_device.value):
                raise RuntimeError('Could not bind TensorRT I/O buffers')
            for name, shape in outputs:
                host = np.empty(shape, dtype=np.float32)
                device = ctypes.c_void_p()
                self._check(self.cuda.cudaMalloc(ctypes.byref(device), host.nbytes),
                            f'cudaMalloc({name})')
                self.output_buffers.append((name, host, device))
                if not self.context.set_tensor_address(name, device.value):
                    raise RuntimeError(f'Could not bind TensorRT output {name}')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f'{operation} failed with CUDA error {status}')

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name, shape=self.input_shape)]

    def get_providers(self) -> list[str]:
        return ['NativeTensorRT']

    def run(self, output_names: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if output_names is not None or set(feeds) != {self.input_name}:
            raise ValueError('Expected the single TensorRT input and all outputs')
        tensor = feeds[self.input_name]
        if tensor.shape != self.input_shape or tensor.dtype != np.float32:
            raise ValueError(f'Expected float32 input of shape {self.input_shape}')
        tensor = np.ascontiguousarray(tensor)
        self._check(self.cuda.cudaMemcpyAsync(
            self.input_device, ctypes.c_void_p(tensor.ctypes.data), tensor.nbytes,
            1, self.stream
        ), 'cudaMemcpy(host-to-device)')
        if not self.context.execute_async_v3(stream_handle=self.stream.value):
            raise RuntimeError('TensorRT execution failed')
        for name, host, device in self.output_buffers:
            self._check(self.cuda.cudaMemcpyAsync(
                ctypes.c_void_p(host.ctypes.data), device,
                host.nbytes, 2, self.stream
            ), f'cudaMemcpy({name} device-to-host)')
        self._check(self.cuda.cudaStreamSynchronize(self.stream),
                    'cudaStreamSynchronize')
        return [host.copy() for _, host, _ in self.output_buffers]

    def close(self) -> None:
        if getattr(self, 'stream', None) is not None and self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
            self.stream.value = None
        pointer = getattr(self, 'input_device', None)
        if pointer is not None and pointer.value:
            self.cuda.cudaFree(pointer)
            pointer.value = None
        for _, _, pointer in getattr(self, 'output_buffers', []):
            if pointer.value:
                self.cuda.cudaFree(pointer)
                pointer.value = None

    def __del__(self) -> None:
        if hasattr(self, 'cuda'):
            self.close()
