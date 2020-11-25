# -*- coding: utf-8 -*-
# author: Yonghao He
# description: 
"""
① convert model to onnx
② generate TRT engine
③ run n loops for evluation
"""
import os
import shutil
import torch
import onnx
import pycuda


def inference_latency_evaluation(model,
                                 input_shapes,
                                 input_names,
                                 output_names,
                                 num_loops):

    temp_onnx_file_path = os.path.join(os.path.dirname(__file__), 'temp.onnx')
    input_tensors = [torch.rand(input_shape) for input_shape in input_shapes]

    print('Start to convert pytorch model to onnx format------------------')
    torch.onnx.export(model,
                      args=tuple(input_tensors),
                      f=temp_onnx_file_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names
                      )
    print('Converting successfully---------------')

    onnx_model = onnx.load(temp_onnx_file_path)
    onnx.checker.check_model(onnx_model)


