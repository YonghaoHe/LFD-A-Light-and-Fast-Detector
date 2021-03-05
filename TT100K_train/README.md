### LFD for TT100K

#### Background
For multi-class detection, we apply LFD to [TT100K[1]](http://cg.cs.tsinghua.edu.cn/traffic-sign/) dataset. In this dataset, there are 45 types of traffic signs used for training,
which is suitable for verify the effectiveness of LFD. We design 2 types of network structures with different sizes of weights and inference latency:
* LFD_L — Large
* LFD_S — Small

These structures can be adopted as templates for your own tasks, or inspire you to create new structures.

#### Performance
We use only train split (6105 images) for model training, and test our models on test split (3071 images). In [1], authors extended the training set: `Classes with
between 100 and 1000 instances in the training set were augmented to give them 1000 instances`. But the augmented data is not released. That means we use much less
data than [1] for training. However, as you can see below, our models can still achieve better performance.

We use the evaluation code release by [1], precision and recall are computed.

Model Version|Precision|Recall
------|--------|----------
**FastRCNN in [1]**|0.5014    |0.5554
**Method proposed in [1]**|0.8773 | 0.9065
**LFD_L**|0.9205 |0.9129 
**LFD_S**|0.9202 |0.9042 
> the score threshold of our models is set to 0.95

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


#### Usage of Files
* [generate_neg_images.py](./generate_neg_images.py) 
    
  Generate pure neg images based on train split of TT100K. The crop rule is simple, just read the code for details.
 
* [pack_tt100k.py](./pack_tt100k.py)
  
  Pack all train data (pos & neg images) as a disk-based dataset.
 
* [timing_inference_latency.py](./timing_inference_latency.py)

  Timing the inference latency of your designed structures without training. 
  Once you finish the config file (like TT100K_LFD_L.py here), you can immediately know how fast it can be.
 
* [TT100K_augmentation_pipeline.py](./TT100K_augmentation_pipeline.py)

  A very simple augmentation pipeline for training and testing, including only data normalization. Based on this script, you can write your own 
  data augmentation pipeline.
 
* [TT100K_LFD_L.py](./TT100K_LFD_L.py) | [TT100K_LFD_S.py](./TT100K_LFD_S.py)
  
  Configure all parameters and **directly run the script for training**.

* [predict.py](./predict.py)

  Make a quick prediction of trained models and qualitative results are shown.
 
* [predict_tensorrt.py](./predict_tensorrt.py)

  Make a quick prediction using tensorrt as inference engine.

* [official_eval.py](./official_eval.py)
  
  Official code for evaluation from [1]. The code was written in Python 2.X, so we modified it to run in Python 3.X.

* [evaluation.py](./evaluation.py)

  Generate result json files and evaluate the results using official evaluation rules.
  
#### Get Started
##### Quick Predict
1. download the pre-trained models and put them in the current folder
2. open the script `predict.py`, and make the following modifications:
    * select the model and import ---- `from TT100K_LFD_S_work_dir_xxxxxxxx_xxxxxx.TT100K_LFD_S import config_dict, prepare_model`
    * set the path of model weight ---- `param_file_path = './TT100K_LFD_S_work_dir_xxxxxxxx_xxxxxx/epoch_500.pth'`
    * set the image path ---- `image_path = './test_images/73.jpg'` (we provide some test images in `./test_images`)
    * set the params of thresholds ---- `classification_threshold=0.5, nms_threshold=0.1, class_agnostic=True`
3. run the script

##### Train with TT100K dataset
1. download the packed dataset or prepare by yourself. Here, we briefly describe the steps, for more information, please refer to the [wiki]():
    * write your own annotation parser for providing samples 
    * pack data as memory-based or disk-based dataset according to your need
2. select a off-the-shelf config script (currently, you have 2 choices----L/S), and directly run the script for training.
An other choice is to write your own config script, including designing new structures. 

#### Download
#### pre-trained models

We provide pre-trained weights of 2 models, as well as training logs, feel free to use them to predict test images. 

* LFD_L pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()
* LFD_S pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()

When successfully download the folder, you just put them in the current fold, namely `./TT100K_train`. It may look like this:
`./TT100K_train/TT100K_LFD_L_work_dir_xxxxxxxx_xxxxxx`.

#### packed TT100K dataset
Download here: [Baidu YunPan](),  [MS OneDrive]()

After you download the packed dataset, you can put it to `./TT100K_pack/train.pkl`.

#### Reference
[1] Zhe Zhu, Dun Liang, Songhai Zhang, Xiaolei Huang, Baoli Li, Shimin Hu, 'Traffic-Sign Detection and Classification in the Wild', CVPR 2016
