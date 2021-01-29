### LFD for TT100K

#### Background
For multi-class detection, we apply LFD to [TT100K[1]](http://cg.cs.tsinghua.edu.cn/traffic-sign/) dataset. In this dataset, there are 45 types of traffic signs used for training.
which is suitable for verify the effectiveness of LFD. We design 2 types of network structures with different sizes of weights and inference latency:
* LFD_L — Large
* LFD_S — Small

These structures can be adopted as templates for your own tasks, or inspire you to create new structures.

#### Performance
We use only train split (6105 images) for model training, and test our models on test split (3071 images). In [1], authors extended the training set: `Classes with
between 100 and 1000 instances in the training set were augmented to give them 1000 instances`. But the augmented data is not released. That means we use much less
data than [1] used for training. However, as you can see below, our models can achieve better performance.

We use the evaluation code release by [1], precision and recall are computed.

Model Version|Precision|Recall
------|--------|----------
**FastRCNN in [1]**|0.5014    |0.5554
**Method proposed in [1]**|0.8773 | 0.9065
**LFD_L**|0.9205 |0.9129 
**LFD_S**|0.9202 |0.9042 
> the score threshold of our models is set to 0.95

##### Inference latency

---
**Platform: RTX 2080Ti, CUDA 10.2, CUDNN 7.6.0.5, TensorRT 7.0.0.11**

* batchsize=1, weight precision mode=FP32

Model Version|1280×720|1920×1080|3840×2160
-------------|-------|-------|--------
**LFD_L**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**LFD_S**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
---
**Platform: RTX 2080Ti, CUDA 10.2, CUDNN 7.6.0.5, TensorRT 7.0.0.11**

* batchsize=1, weight precision mode=FP32

Model Version|1280×720|1920×1080|3840×2160
-------------|-------|-------|--------
**LFD_L**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)
**LFD_S**|-ms(-FPS)|-ms(-FPS)|-ms(-FPS)


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
  
  Configure all parameters and run the script for training.

* [predict.py](./predict.py)

  Make a quick prediction of trained models and qualitative results are shown.

* [official_eval.py](./official_eval.py)
  
  Official code for evaluation from [1]. The code was written in Python 2.X, so we modified it to run in Python 3.X.

* [evaluation.py](./evaluation.py)

  Generate result json files and evaluate the results using Presion and Recall metrics.
  

#### Download
We provide pre-trained weights of 2 models, feel free to use them to predict test images. We do not provide packed dataset, you can use the relevant 
scripts to generate the disk-based dataset.

* LFD_L pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()
* LFD_S pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()

#### Reference
[1] Zhe Zhu, Dun Liang, Songhai Zhang, Xiaolei Huang, Baoli Li, Shimin Hu, 'Traffic-Sign Detection and Classification in the Wild', CVPR 2016
