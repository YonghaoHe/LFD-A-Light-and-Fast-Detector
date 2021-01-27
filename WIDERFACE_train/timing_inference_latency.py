# -*- coding: utf-8 -*-

from lfd.deployment.tensorrt.build_engine import GB
from lfd.deployment.tensorrt.inference_latency_evaluation import inference_latency_evaluation

# set model
from WIDERFACE_LFD_XS import config_dict, prepare_model

prepare_model()

inference_latency_evaluation(
    model=config_dict['model'],
    input_shapes=[[1, 3, 720, 1280]],
    input_names=['input_data'],
    output_names=['classification_output', 'regression_output'],
    precision_mode='fp16',
    max_workspace_size=GB(6),
    min_timing_iterations=2,
    avg_timing_iterations=2,
    int8_calibrator=None,
    timing_loops=1000
)
