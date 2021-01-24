# This repo is still under construction! Do not clone currently!

## Update History

## 1. Introduction
In this repo, we release a new One-Stage Anchor-Free Detector called **LFD**. The new LFD completely surpasses the previous 
**[LFFD](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices)** in most aspects. We are trying
to make object detection easier, explainable and more applicable. With LFD, you are able to train and deploy a desired model 
without all the bells and whistles.

### 1.1 Highlights
Compared to LFFD, LFD has the following features:
* implemented with PyTorch, which is friendly for most people
* support multi-class rather than single-class
* higher precision and lower inference latency
* we maintain a [wiki]()(highly recommend) for you to fully master the code
* the performance of LFD has been proved in more real-world applications
* train from scratch on your own datasets, create your desired models with satisfactory model size and inference latency

### 1.2 Sneak Peek
Before dive into the code, we present some performance results on several datasets in the beginning, 
including precision and inference latency.

#### Dataset 1: WIDERFACE (Single-class)
##### Accuracy on val under the **SIO** schema proposed in [LFFD](https://arxiv.org/abs/1904.10633)

Model Version|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
**[v2](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/tree/master/face_detection)**|0.875     |0.863       |0.754
**WIDERFACE-L**| - | - | -
**WIDERFACE-M**| - | - | -
**WIDERFACE-S**| - | - | -
**WIDERFACE-XS**| - | - | -

> * for fairy comparison, WIDERFACE-L/M/S/XS have the similar detection range with v2, namely [4, 320] vs [10, 320], but WIDERFACE-L/M/S/XS have 
different network structures.
> * great improvement on Hard Set

##### Inference latency on different resolutions and GPUs

**Platform: RTX 2080Ti, CUDA 10.2, CUDNN 7.6.0.5, TensorRT 7.0.0.11**

* batchsize=1, weight precision mode=FP32

Model Version|320×240|640×480|1280×720|1920×1080|3840×2160|7680×4320
-------------|-------|-------|--------|---------|---------|---------
**v2**|1.06ms(946.04FPS)|2.12ms(472.04FPS)|5.02ms(199.10FPS)|10.80ms(92.63FPS)|42.41ms(23.58FPS)|167.25ms(5.98FPS)
**WIDERFACE-L**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-M**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-S**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-XS**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)

> the results of v2 is directly get from [LFFD](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/tree/master/face_detection),
the Platform condition is slightly different from here.

* batchsize=8, weight precision mode=FP16

Model Version|320×240|640×480|1280×720|1920×1080|3840×2160|7680×4320
-------------|-------|-------|--------|---------|---------|---------
**WIDERFACE-L**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-M**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-S**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**WIDERFACE-XS**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)

> Ultra high throughput !

#### Dataset 2: xxxxxxxxxx (Multi-class)

## 2. Get Started

### 2.1 Install Procedure

**Prerequirements**  
* python = 3.6
* albumentations = 0.4.6
* torch = 1.4
* torchvision = 0.5.0
* cv2 = 4.0
* numpy = 1.16
* pycocotools = 2.0.1

> All above versions are employed, other versions may work as well.

**Build Internal Libs**

In the repo root, run the code below:

`python setup.py build_ext`

Once successful, you will see: `----> build and copy successfully!`
> if you want to know what libs are built and where they are copied, you can read the file setup.py.

**Build External Libs**
* Build libjpeg-turbo
  1. download the [source code v2.0.5](https://sourceforge.net/projects/libjpeg-turbo/files/)
  2. decompress and compile:
     > `cd [source code]`  
       `mkdir build`  
       `cd build`  
       `cmake ..`  
       `make`
      
     > make sure that `cmake` configuration properly
  3. copy `build/libturbojpeg.so.x.x.x` to `lfd/data_pipeline/dataset/utils/libs`

**Add PYTHONPATH**

The last step is to add the repo root to PYTHONPATH. You have two ways:
* permanent way: append `export PYTHONPATH=[repo root]:$PYTHONPATH` to the file ~/.bashrc
* temporal way: whenever you want to code with the repo, add the following code ahead:
  1. `import sys`
  2. `sys.path.append('path to the repo')`
 
Until now, the repo is ready for use. By the way, we do not install the repo to the default python libs location 
(like /python3.x/site-packages/) for easily modification and development.

### 2.2 Prepare the Dataset

### 2.3 Train

### 2.4 Deploy

## 3. Advanced Guide

## 4. Q&A

## Acknowledgement
* very much thankful for [PyTorch](https://pytorch.org/) framework.
* we learn a lot and reuse some basic code from [mmdetection](https://github.com/open-mmlab/mmdetection), thanks for the great work.
* thanks for some third-party libs like [albumentations](https://github.com/albumentations-team/albumentations), [turbojpeg](https://libjpeg-turbo.org/).

     
     