# -*- coding: utf-8 -*-

"""
① convert model to onnx
② generate TRT engine
③ run n loops for evaluation
"""
import os
import numpy
import time
import torch
import onnx
import pycuda.driver as cuda
import tensorrt
from .build_engine import build_tensorrt_engine, GB
from .inference import allocate_buffers


def timing_engine(engine_file_path,
                  batch_size,
                  num_input_channels,
                  height,
                  width,
                  timing_loops=100):
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)

    with open(engine_file_path, 'rb') as fin, tensorrt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(fin.read())

    assert engine is not None, 'deserialize engine failed!'
    assert batch_size <= engine.max_batch_size

    print('Engine info:')
    print('\tmax batch size: ', engine.max_batch_size)
    print('\tmax workspace_size: ', engine.max_workspace_size)
    print('\tdevice memory_size: ', engine.device_memory_size)

    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size)

    input_data = numpy.random.rand(batch_size, num_input_channels, height, width).astype(dtype=numpy.float32, order='C')
    inputs[0].host = input_data

    print('Start timing......')

    with engine.create_execution_context() as context:

        # warm up
        for i in range(10):
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            stream.synchronize()

        time_start = time.time()
        for i in range(timing_loops):
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            stream.synchronize()
        time_end = time.time()

        print('Total time elapsed: %.04f ms.\n%.04f ms for each image (%.02f FPS)\n%.04f ms for each batch' %
              ((time_end - time_start) * 1000,
               (time_end - time_start) * 1000 / batch_size / timing_loops,
               batch_size * timing_loops / (time_end - time_start),
               (time_end - time_start) * 1000 / timing_loops))


def inference_latency_evaluation(model,
                                 input_shapes,
                                 input_names,
                                 output_names,
                                 precision_mode='fp32',
                                 max_workspace_size=GB(1),
                                 min_timing_iterations=2,
                                 avg_timing_iterations=2,
                                 int8_calibrator=None,
                                 timing_loops=100):
    '''

    :param model: the net to be evaluated (torch.Module)
    :param input_shapes: list of input shapes in [N, C, H, W], e.g., [[1, 3, 480, 640]]
    :param input_names: list of input names, e.g., ['input_data']
    :param output_names: list of output names, the number of names should be the same as that of model
    :param precision_mode: choose from 'fp32', 'fp16', 'int8'
    :param max_workspace_size:
    :param min_timing_iterations:
    :param avg_timing_iterations:
    :param int8_calibrator:
    :param timing_loops:
    :return:
    '''
    temp_onnx_file_path = os.path.join(os.path.dirname(__file__), 'temp', 'temp.onnx')
    if not os.path.exists(os.path.dirname(temp_onnx_file_path)):
        os.makedirs(os.path.dirname(temp_onnx_file_path))

    input_tensors = [torch.rand(input_shape) for input_shape in input_shapes]

    print('Start to convert pytorch model to onnx format------------------')
    torch.onnx.export(model,
                      args=tuple(input_tensors),
                      f=temp_onnx_file_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=9
                      )
    print('Converting successfully---------------')

    onnx_model = onnx.load(temp_onnx_file_path)
    onnx.checker.check_model(onnx_model)

    temp_engine_save_path = os.path.join(os.path.dirname(__file__), 'temp', 'temp.engine')
    if build_tensorrt_engine(temp_onnx_file_path,
                             temp_engine_save_path,
                             precision_mode=precision_mode,
                             max_workspace_size=max_workspace_size,  # in bytes
                             max_batch_size=input_shapes[0][0],
                             min_timing_iterations=min_timing_iterations,
                             avg_timing_iterations=avg_timing_iterations,
                             int8_calibrator=int8_calibrator):
        timing_engine(engine_file_path=temp_engine_save_path,
                      batch_size=input_shapes[0][0],
                      num_input_channels=input_shapes[0][1],
                      height=input_shapes[0][2],
                      width=input_shapes[0][3],
                      timing_loops=timing_loops)
