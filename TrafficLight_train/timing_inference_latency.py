# -*- coding: utf-8 -*-
import os
import numpy
from lfd.deployment.tensorrt.build_engine import GB, INT8Calibrator
from lfd.deployment.tensorrt.inference_latency_evaluation import inference_latency_evaluation

# get model
from TL_LFD_S import config_dict, prepare_model

prepare_model()

# seting params
input_shapes = [[1, 3, 720, 1280]]
precision_mode = 'fp32'
max_workspace_size = GB(6)
min_timing_iterations = 2
avg_timing_iterations = 2
timing_loops = 1000
engine_root = './tensorrt_engine_folder'

if not os.path.exists(engine_root):
    os.makedirs(engine_root)

if precision_mode == 'int8':
    # construct int8 calibrator
    data_batch = numpy.random.rand(128, 3, 512, 512).astype(numpy.float32)  # use a fake batch
    int8_calibrator = INT8Calibrator(data_batch, os.path.join(engine_root, 'int8_calibration.cache'), batch_size=8)

else:
    int8_calibrator = None

inference_latency_evaluation(
    model=config_dict['model'],
    input_shapes=input_shapes,
    input_names=['input_data'],
    output_names=['classification_output', 'regression_output'],
    precision_mode=precision_mode,
    max_workspace_size=max_workspace_size,
    min_timing_iterations=min_timing_iterations,
    avg_timing_iterations=avg_timing_iterations,
    int8_calibrator=int8_calibrator,
    timing_loops=timing_loops
)

if os.path.exists(os.path.join(engine_root, 'int8_calibration.cache')):
    os.remove(os.path.join(engine_root, 'int8_calibration.cache'))
