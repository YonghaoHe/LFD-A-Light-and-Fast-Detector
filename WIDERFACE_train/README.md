### LFD for WIDERFACE

#### Background
For single-class detection, we apply LFD to [WIDERFACE](http://shuoyang1213.me/WIDERFACE/) dataset which is large and diverse.
We design 4 types of network structures with different sizes of weights and inference latency:
* LFD_L — Large
* LFD_M — Medium
* LFD_S — Small
* LFD_XS — Extreme Small

These structures can be adopted as templates for your own tasks, or inspire you to create new structures.


#### Performance


#### Usage of Files
* [generate_neg_images.py](./generate_neg_images.py) 
    
  Generate pure neg images based on train set of WIDERFACE. The crop rule is simple, just read the code for details.
 
* [pack_widerface.py](./pack_widerface.py)
  
  Pack all train data (pos & neg images) as a memory-based dataset.

* [timing_inference_latency.py](./timing_inference_latency.py)

  Timing the inference latency of your designed structures without training. 
  Once you finish the config file (like WIDERFACE_LFD_L.py here), you can immediately know how fast it can be.
 
* [WIDERFACE_LFD_L.py](./WIDERFACE_LFD_L.py) | [WIDERFACE_LFD_M.py](./WIDERFACE_LFD_M.py) | [WIDERFACE_LFD_S.py](WIDERFACE_LFD_S.py) |
  [WIDERFACE_LFD_XS.py](./WIDERFACE_LFD_XS.py)
  
  Configure all parameters and run the script for training.

* [predict.py](./predict.py)

  Make a quick prediction of trained models and qualitative results are shown.

* [evaluation.py](./evaluation.py)

  Generate files of SIO results for WIDERFACE standard evaluation. You have to use [matlab code](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)
  to get final evaluation metrics. 

#### Download 
We provide pre-trained weights of 4 models, feel free to try. As for the packed dataset, we advise you to pack by yourself. On one hand, 
you can know how to pack a memory-based dataset, on the other hand, you will experience the complete workflow.

* LFD_L pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()
* LFD_M pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()
* LFD_S pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()
* LFD_XS pre-trained weight: [Baidu YunPan](),  [MS OneDrive]()

When successfully download the folder, you just put them in the current fold, namely ``./WIDERFACE_train``. It may look like this:
``./WIDERFACE_train/WIDERFACE_LFD_L_work_dir_xxxxxxxx_xxxxxx``.

