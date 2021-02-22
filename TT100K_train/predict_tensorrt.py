# -*- coding: utf-8 -*-
# predict using tensorrt as inference engine
import torch
import onnx
import os
from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
from lfd.deployment.tensorrt.inference import allocate_buffers
from lfd.deployment.tensorrt.build_engine import GB, build_tensorrt_engine
import tensorrt

tensorrt.init_libnvinfer_plugins(None, '')
import cv2

# set the target model script
from TT100K_LFD_L_work_dir_20210126_163013.TT100K_LFD_L import config_dict, prepare_model

prepare_model()

# set the model weight file
param_file_path = './TT100K_LFD_L_work_dir_20210126_163013/epoch_500.pth'

load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

# set the image path to be tested
image_path = '/home/yonghaohe/datasets/TT100K/data/test/138.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# generate engine
engine_folder = './tensorrt_engine_folder'
if not os.path.exists(engine_folder):
    os.makedirs(engine_folder)

input_shapes = [[1, 3, image.shape[0], image.shape[1]]]
input_names = ['input_data']
output_names = ['classification_output', 'regression_output']
precision_mode = 'fp16'
max_workspace_size = GB(6)
min_timing_iterations = 2
avg_timing_iterations = 2

onnx_file_path = os.path.join(engine_folder, param_file_path.split('/')[-2] + '_' + param_file_path.split('/')[-1].split('.')[0] + '_' + precision_mode + '.onnx')
engine_file_path = os.path.join(engine_folder, param_file_path.split('/')[-2] + '_' + param_file_path.split('/')[-1].split('.')[0] + '_' + precision_mode + '.engine')

if not os.path.exists(engine_file_path):
    # generate onnx file
    input_tensors = [torch.rand(input_shape) for input_shape in input_shapes]
    torch.onnx.export(config_dict['model'],
                      args=tuple(input_tensors),
                      f=onnx_file_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=9
                      )
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    assert build_tensorrt_engine(onnx_file_path,
                                 engine_file_path,
                                 precision_mode=precision_mode,
                                 max_workspace_size=max_workspace_size,  # in bytes
                                 max_batch_size=input_shapes[0][0],
                                 min_timing_iterations=min_timing_iterations,
                                 avg_timing_iterations=avg_timing_iterations,
                                 int8_calibrator=None)

# inference
logger = tensorrt.Logger(tensorrt.Logger.ERROR)
with open(engine_file_path, 'rb') as fin, tensorrt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(fin.read())

assert engine is not None, 'deserialize engine failed!'
print('Engine info:')
print('\tmax batch size: ', engine.max_batch_size)
print('\tmax workspace_size: ', engine.max_workspace_size)
print('\tdevice memory_size: ', engine.device_memory_size)

inputs, outputs, bindings, stream = allocate_buffers(engine, 1)

engine_context = engine.create_execution_context()

results = config_dict['model'].predict_for_single_image_with_tensorrt(image,
                                                                      input_buffers=inputs,
                                                                      output_buffers=outputs,
                                                                      bindings=bindings,
                                                                      stream=stream,
                                                                      engine=engine,
                                                                      tensorrt_engine_context=engine_context,
                                                                      aug_pipeline=simple_widerface_val_pipeline,
                                                                      classification_threshold=0.5,
                                                                      nms_threshold=0.1,
                                                                      class_agnostic=True)

for bbox in results:
    print(bbox)
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 1)
print('%d traffic signs are detected!' % len(results))
cv2.imshow('im', image)
cv2.waitKey()
