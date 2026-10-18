"""
TensorRT Engine wrapper for inference
Handles engine loading and execution
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRT_Engine:
    """
    Wrapper class for TensorRT engine inference
    """
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.outputs = {}
        self.inputs = {}
        self.stream = None
        self.load_engine()

    def load_engine(self):
        """Load TensorRT engine from file and setup I/O buffers"""
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        num_io_tensors = self.engine.num_io_tensors
        
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            size = trt.volume(shape)
            if size <= 0:
                size = 16
            
            d_buffer = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            h_buffer = cuda.pagelocked_empty(size, dtype=dtype)
            self.context.set_tensor_address(name, int(d_buffer))
            
            info = {
                'host': h_buffer,
                'device': d_buffer,
                'shape': shape,
                'dtype': dtype,
                'name': name
            }
            
            if is_input:
                self.inputs[name] = info
            else:
                self.outputs[name] = info

    def inference(self, input_data: dict) -> dict:
        """
        Run inference on the engine
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Copy input data to device
        for name, data in input_data.items():
            if name not in self.inputs:
                continue
            
            data_shape = data.shape
            try:
                current_shape = self.context.get_tensor_shape(name)
                if tuple(current_shape) != tuple(data_shape):
                    self.context.set_input_shape(name, data_shape)
            except:
                pass
            
            info = self.inputs[name]
            if data.dtype != info['dtype']:
                data = data.astype(info['dtype'])
            
            data_flat = data.ravel()
            np.copyto(info['host'][:len(data_flat)], data_flat)

        for name in input_data.keys():
            if name in self.inputs:
                info = self.inputs[name]
                cuda.memcpy_htod_async(info['device'], info['host'], self.stream)

        # Execute inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT execution failed")

        # Copy output data to host
        for info in self.outputs.values():
            cuda.memcpy_dtoh_async(info['host'], info['device'], self.stream)
        
        self.stream.synchronize()

        # Reshape outputs
        output_results = {}
        for name, info in self.outputs.items():
            try:
                output_shape = self.context.get_tensor_shape(name)
            except:
                output_shape = info['shape']
            
            vol = trt.volume(output_shape)
            output_results[name] = info['host'][:vol].reshape(output_shape)
            
        return output_results