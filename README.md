## Update History
* `2021.03.30` We maintain an [Experiments wiki page](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector/wiki/Experiments) to show ablation studies for your reference. Maybe these experiments are valuable for you to make proper
decisions.
* `2021.03.16` INT8 inference is updated. Check [timing_inference_latency.py](./WIDERFACE_train/timing_inference_latency.py) and [predict_tensorrt.py](./WIDERFACE_train/predict_tensorrt.py) for reference.
* `2021.03.09` LFD now is formally released!!! Any questions and problems are welcome.

## 1. Introduction
In this repo, we release a new One-Stage Anchor-Free Detector named **LFD**. LFD completely surpasses the previous 
**[LFFD](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices)** in most aspects. We are trying
to make object detection easier, explainable and more applicable. With LFD, you are able to train and deploy a desired model 
without all the bells and whistles. Eventually, we hope LFD can be as popular as YOLO series for the industrial community in the future.

### 1.1 New Features
Compared to LFFD, LFD has the following features:
* implemented using PyTorch, which is friendly for most guys (LFFD is implemented using MXNet)
* support multi-class detection rather than single-class detection (LFFD is only for single-class)
* higher precision and lower inference latency
* we maintain a [wiki](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector/wiki) (highly recommended) for you to fully understand LFD and master the code
* the performance of LFD has been proved in many real-world applications
* create your desired models with satisfactory model size and inference latency, train from scratch on your own datasets, 

### 1.2 Performance Highlights
Before dive into the code, we present some performance results on two datasets, 
including precision and inference latency.

#### Dataset 1: WIDERFACE (single-class)
##### Accuracy on val under the **SIO** evaluation schema proposed in [LFFD](https://arxiv.org/abs/1904.10633)

Model Version|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
**[v2](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/tree/master/face_detection)**|0.875     |0.863       |0.754
**WIDERFACE-L**|0.887 |0.896 |0.863
**WIDERFACE-M**|0.874 |0.888 |0.855
**WIDERFACE-S**|0.873 |0.885 |0.849
**WIDERFACE-XS**|0.866 |0.877 |0.839

> * v2 is from LFFD, you can check it in [LFFD repo](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/tree/master/face_detection).
> * for fairy comparison, WIDERFACE-L/M/S/XS have the similar detection range with v2, namely [4, 320] vs [10, 320], but WIDERFACE-L/M/S/XS have 
different network structures.
> * great improvement on Hard Set.

##### Inference latency

**Platform: RTX 2080Ti, CUDA 10.2, CUDNN 8.0.4, TensorRT 7.2.2.3**

* batchsize=1, weight precision mode=FP32

Model Version|640×480|1280×720|1920×1080|3840×2160
-------------|-------|--------|---------|---------
**v2**|2.12ms(472.04FPS)|5.02ms(199.10FPS)|10.80ms(92.63FPS)|42.41ms(23.58FPS)
**WIDERFACE-L**|2.67ms(374.19FPS)|6.31ms(158.38FPS)|13.51ms(74.04FPS)|94.61ms(10.57FPS)
**WIDERFACE-M**|2.47ms(404.23FPS)|5.70ms(175.38FPS)|12.28ms(81.43FPS)|87.90ms(11.38FPS)
**WIDERFACE-S**|1.82ms(548.42FPS)|3.57ms(280.00FPS)|7.35ms(136.02FPS)|27.93ms(35.81FPS)
**WIDERFACE-XS**|1.58ms(633.06FPS)|3.03ms(330.36FPS)|6.14ms(163.00FPS)|23.26ms(43.00FPS)

> the results of v2 is directly get from [LFFD](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/tree/master/face_detection),
the Platform condition is slightly different from here.

* batchsize=1, weight precision mode=FP16

Model Version|640×480|1280×720|1920×1080|3840×2160
-------------|-------|--------|---------|---------
**WIDERFACE-L**|1.68ms(594.12FPS)|3.69ms(270.78FPS)|7.66ms(130.51FPS)|28.65ms(34.90FPS)
**WIDERFACE-M**|1.61ms(622.42FPS)|3.51ms(285.13FPS)|7.31ms(136.79FPS)|27.32ms(36.60FPS)
**WIDERFACE-S**|1.26ms(793.97FPS)|2.39ms(418.68FPS)|4.88ms(205.09FPS)|18.46ms(54.18FPS)
**WIDERFACE-XS**|1.23ms(813.01FPS)|2.18ms(459.17FPS)|4.57ms(218.62FPS)|17.35ms(57.65FPS)

> It can be observed that FP16 mode is evidently faster than FP32 mode. So in deployment, FP16 is highly recommended if possible.

* batchsize=1, weight precision mode=INT8

Model Version|640×480|1280×720|1920×1080|3840×2160
-------------|-------|--------|---------|---------
**WIDERFACE-L**|1.50ms(667.95FPS)|3.24ms(308.43FPS)|6.83ms(146.41FPS)|-ms(-FPS)
**WIDERFACE-M**|1.45ms(689.00FPS)|3.15ms(317.60FPS)|6.61ms(151.20FPS)|-ms(-FPS)
**WIDERFACE-S**|1.17ms(855.29FPS)|2.14ms(466.86FPS)|4.40ms(227.18FPS)|-ms(-FPS)
**WIDERFACE-XS**|1.09ms(920.91FPS)|2.03ms(493.54FPS)|4.11ms(243.15FPS)|-ms(-FPS)

> CAUTION: `-` means results are not available due to out of memory while calibrating

#### Dataset 2: TT100K (multi-class----45 classes)
##### Precision&Recall on test set of [TT100K[1]](http://cg.cs.tsinghua.edu.cn/traffic-sign/)

Model Version|Precision|Recall
------|--------|----------
**FastRCNN in [1]**|0.5014    |0.5554
**Method proposed in [1]**|0.8773 | 0.9065
**LFD_L**|0.9205 |0.9129 
**LFD_S**|0.9202 |0.9042 
> We use only train split (6105 images) for model training, and test our models on test split (3071 images). In [1], authors extended the training set: `Classes with
between 100 and 1000 instances in the training set were augmented to give them 1000 instances`. But the augmented data is not released. That means we use much less
data than [1] used for training. However, as you can see, our models can still achieve better performance.
Precision&Recall results of [1] can be found in it's released code folder: `code/results/report_xxxx.txt`.

##### Inference latency

**Platform: RTX 2080Ti, CUDA 10.2, CUDNN 8.0.4, TensorRT 7.2.2.3**

* batchsize=1, weight precision mode=FP32

Model Version|1280×720|1920×1080|3840×2160
-------------|-------|-------|--------
**LFD_L**|9.87ms(101.35FPS)|21.56ms(46.38FPS)|166.66ms(6.00FPS)
**LFD_S**|4.31ms(232.27FPS)|8.96ms(111.64FPS)|34.01ms(29.36FPS)

* batchsize=1, weight precision mode=FP16

Model Version|1280×720|1920×1080|3840×2160
-------------|-------|-------|--------
**LFD_L**|6.28ms(159.27FPS)|13.09ms(76.38FPS)|49.79ms(20.09FPS)
**LFD_S**|3.03ms(329.68FPS)|6.27ms(159.54FPS)|23.41ms(42.72FPS)

* batchsize=1, weight precision mode=INT8

Model Version|1280×720|1920×1080|3840×2160
-------------|-------|-------|--------
**LFD_L**|5.96ms(167.89FPS)|12.68ms(78.86FPS)|-ms(-FPS)
**LFD_S**|2.90ms(345.33FPS)|5.89ms(169.86FPS)|-ms(-FPS)

> CAUTION: `-` means results are not available due to out of memory while calibrating

## 2. Get Started

### 2.1 Install

**Prerequirements**  
* python => 3.6
* albumentations => 0.4.6
* torch => 1.5
* torchvision => 0.6.0
* cv2 => 4.0
* numpy => 1.16
* pycocotools => 2.0.1
* pycuda = 2020.1
* tensorrt = 7.2.2.3 (corresponding cudnn = 8.0)

> All above versions are tested, newer versions may work as well but not fully tested.

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

The last step is to add the repo root to PYTHONPATH. You have two choices:
1. permanent way: append `export PYTHONPATH=[repo root]:$PYTHONPATH` to the file ~/.bashrc
2. temporal way: whenever you want to code with the repo, add the following code ahead:
   1. `import sys`
   2. `sys.path.append('path to the repo')`
 
Until now, the repo is ready for use. By the way, we do not install the repo to the default python libs location 
(like /python3.x/site-packages/) for easily modification and development.

**Docker Installation**

please check [here](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector/issues/1) for more details, thanks to [@ashuezy](https://github.com/ashuezy).

### 2.2 Play with the code
We present the details of how to use the code in two specific tasks.
* face detection on WIDERFACE [README.md](./WIDERFACE_train/README.md)
* traffic sign detection on TT100K [README.md](./TT100K_train/README.md)

Besides, we describe the structure of code in [wiki](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector/wiki).

## Acknowledgement
* very much thankful for [PyTorch](https://pytorch.org/) framework.
* we learn a lot and reuse some basic code from [mmdetection](https://github.com/open-mmlab/mmdetection), thanks for the great work.
* thanks for some third-party libs like [albumentations](https://github.com/albumentations-team/albumentations), [turbojpeg](https://libjpeg-turbo.org/).

## Citation
If you find the repo is useful, please cite the repo website directly.
     