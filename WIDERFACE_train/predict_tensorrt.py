# -*- coding: utf-8 -*-
# predict using tensorrt as inference engine

# In this script, we provide you a demo only.
# In this demo, we build engines that accepts fixed input shape (tensorrt supports dynamic shape, and we will implement it later)
# If you want to deploy in the product, you should make some modifications, like rewrite the post-processing, in order to make it more efficient
import numpy
import torch
import onnx
import os
from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
from lfd.deployment.tensorrt.inference import allocate_buffers
from lfd.deployment.tensorrt.build_engine import GB, build_tensorrt_engine, INT8Calibrator
from lfd.data_pipeline.sampler.region_sampler import crop_from_image
import tensorrt
tensorrt.init_libnvinfer_plugins(None, '')
import cv2


def prepare_data_for_int8_calibrator(image_root, normalizer, crop_size=(512, 512)):
    """
    Steps:
    1.    read images and crop images
    2.    do normalization (if you have integrated normalization in computation graph, then normalization can be abandoned)
    3.    organize all images into one batch
    :param image_root: path that contains all images for calibration (images from test set that cover most situations, and the number of images is not necessarily large, say 128 is enough)
    :param normalizer: normalizer used for normalizing images
    :param crop_size: (w, h)
    :return:
    """
    assert os.path.exists(image_root), '[%s] root does not exist!!!'
    image_paths_list = [os.path.join(image_root, name) for name in os.listdir(image_root) if name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    batch = numpy.zeros((len(image_paths_list), 3, crop_size[1], crop_size[0]), dtype=numpy.float32)

    for i in range(len(image_paths_list)):
        image = cv2.imread(image_paths_list[i], cv2.IMREAD_COLOR)
        image_crop = crop_from_image(image, (int((image.shape[1] - crop_size[0]) / 2), int((image.shape[0] - crop_size[1]) / 2), crop_size[0], crop_size[1]))
        image_crop_normalized = normalizer(**{'image': image_crop})['image']

        batch[i] = image_crop_normalized.transpose([2, 0, 1])

    return batch


# set the target model script ------------------------------------------------------------------------
from WIDERFACE_LFD_XS_work_dir_20210210_115210.WIDERFACE_LFD_XS import config_dict, prepare_model

prepare_model()

# set the model weight file ------------------------------------------------------------------------
param_file_path = './WIDERFACE_LFD_XS_work_dir_20210210_115210/epoch_1000.pth'

load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

# set the image path to be tested ------------------------------------------------------------------------
image_path = './test_images/image1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# generate engine ------------------------------------------------------------------------
engine_folder = './tensorrt_engine_folder'
if not os.path.exists(engine_folder):
    os.makedirs(engine_folder)

# setting for engine building
input_shapes = [[1, 3, image.shape[0], image.shape[1]]]
input_names = ['input_data']
output_names = ['classification_output', 'regression_output']
precision_mode = 'int8'
max_workspace_size = GB(6)
min_timing_iterations = 2
avg_timing_iterations = 2

int8_calibrator = None
image_root_for_int8_calibration = '/home/yonghaohe/datasets/WIDER_FACE/WIDER_val/images/0--Parade'
normalizer = simple_normalize

if precision_mode == 'int8':
    data_batch = prepare_data_for_int8_calibrator(image_root=image_root_for_int8_calibration, normalizer=normalizer, crop_size=(512, 512))
    int8_calibrator = INT8Calibrator(data_batch, os.path.join(engine_folder, param_file_path.split('/')[-2] + '_' + param_file_path.split('/')[-1].split('.')[0] + '_' + precision_mode + '.cache'), batch_size=8)
else:
    int8_calibrator = None


onnx_file_path = os.path.join(engine_folder, param_file_path.split('/')[-2] + '_' + param_file_path.split('/')[-1].split('.')[0] + '_' + precision_mode + '.onnx')
engine_file_path = os.path.join(engine_folder, param_file_path.split('/')[-2] + '_' + param_file_path.split('/')[-1].split('.')[0] + '_' + precision_mode + '.engine')

# if the engine exists, skip building process
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
                                 int8_calibrator=int8_calibrator)

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
                                                                      nms_threshold=0.3,
                                                                      class_agnostic=False)

for bbox in results:
    print(bbox)
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 1)
print('%d faces are detected!' % len(results))
cv2.imshow('im', image)
cv2.waitKey()
