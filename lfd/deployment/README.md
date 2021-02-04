### Deployment
Currently, we support NVIDIA TensorRT deployment.

### Prerequirements
* pycuda = 2020.1
* tensorrt = 7.2.2.3 (corresponding cudnn = 8.0)

> GroupNorm is supported in TensorRT 7.2.X and newer. If you use GN, make sure the version of TensorRT meets the requirement.