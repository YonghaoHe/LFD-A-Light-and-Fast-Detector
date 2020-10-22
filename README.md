## This repo is still under construction! Do not clone currently!


## Get Started

### Install Tips

**Prerequirements**  
* python = 3.6
* albumentations = 0.4.6
* torch = 1.4
× torchvision = 0.5.0
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
  3. copy `build/libturbojpeg.so.x.x.x` to `data_pipeline/dataset/utils/libs`

**Data Preparation**

     
     