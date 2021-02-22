# -*- coding: utf-8 -*-

import os
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt

__all__ = ['MB', 'GB', 'build_tensorrt_engine']

EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def MB(val):
    return val * 1 << 20


def GB(val):
    return val * 1 << 30


class INT8Calibrator(tensorrt.IInt8EntropyCalibrator2):

    def __init__(self, data, cache_file, batch_size=8):
        """

        :param data: numpy array with shape (N, C, H, W)
        :param cache_file:
        :param batch_size:
        """
        tensorrt.IInt8EntropyCalibrator2.__init__(self)

        self._cache_file = cache_file
        self._batch_size = batch_size

        """
        data is numpy array in float32, caution: each image should be normalized
        """
        assert data.ndim == 4 and data.dtype == numpy.float32
        self._data = numpy.array(data, dtype=numpy.float32, order='C')

        self._current_index = 0

        # Allocate enough memory for a whole batch.
        self._device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names, p_str=None):
        if self._current_index + self._batch_size > self._data.shape[0]:
            return None

        current_batch = int(self._current_index / self._batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self._batch_size))

        batch = self._data[self._current_index:self._current_index + self._batch_size]
        cuda.memcpy_htod(self._device_input, batch)
        self._current_index += self._batch_size
        return [self._device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self._cache_file, "wb") as f:
            f.write(cache)


def build_tensorrt_engine(onnx_file_path,
                          engine_save_path,
                          precision_mode='fp32',
                          max_workspace_size=GB(1),  # in bytes
                          max_batch_size=1,
                          min_timing_iterations=2,
                          avg_timing_iterations=2,
                          int8_calibrator=None):
    """

    :param onnx_file_path:
    :param engine_save_path:
    :param precision_mode:
    :param max_workspace_size: The maximum workspace size. The maximum GPU temporary memory which the engine can use at
    :param max_batch_size:
    :param min_timing_iterations:
    :param avg_timing_iterations:
    :param int8_calibrator:
    :return:
    """
    assert os.path.exists(onnx_file_path)
    assert precision_mode in ['fp32', 'fp16', 'int8']

    trt_logger = tensorrt.Logger(tensorrt.Logger.VERBOSE)

    builder = tensorrt.Builder(trt_logger)
    if precision_mode == 'fp16':
        assert builder.platform_has_fast_fp16, 'platform does not support fp16 mode!'
    if precision_mode == 'int8':
        assert builder.platform_has_fast_int8, 'platform does not support int8 mode!'
        assert int8_calibrator is not None, 'calibrator is not provided!'

    network = builder.create_network(EXPLICIT_BATCH)

    parser = tensorrt.OnnxParser(network, trt_logger)

    with open(onnx_file_path, 'rb') as onnx_fin:
        parser.parse(onnx_fin.read())

    num_error = parser.num_errors
    if num_error != 0:
        for i in range(num_error):
            temp_error = parser.get_error(i)
            print(temp_error.desc())
        return

    config = builder.create_builder_config()

    if precision_mode == 'int8':
        config.int8_calibrator = int8_calibrator
        config.set_flag(tensorrt.BuilderFlag.INT8)
    elif precision_mode == 'fp16':
        config.set_flag(tensorrt.BuilderFlag.FP16)
    else:
        pass

    config.max_workspace_size = max_workspace_size
    config.min_timing_iterations = min_timing_iterations
    config.avg_timing_iterations = avg_timing_iterations
    builder.max_batch_size = max_batch_size
    try:
        engine = builder.build_engine(network, config)
    except:
        print('Engine build unsuccessfully!')
        return False

    if engine is None:
        print('Engine build unsuccessfully!')
        return False

    if not os.path.exists(os.path.dirname(engine_save_path)):
        os.makedirs(os.path.dirname(engine_save_path))

    serialized_engine = engine.serialize()
    with open(engine_save_path, 'wb') as fout:
        fout.write(serialized_engine)

    print('Engine built successfully!')
    return True
