### Usage

* Supported OS: the source code was tested on 64-bit Arch and Ubuntu 14.04 Linux OS, and it should also be executable in other linux distributions.

* Dependencies: 
 * A modified version of [caffe](http://caffe.berkeleyvision.org/) framework and all its dependencies. 
 * Cuda enabled GPUs

* Installation: 
 1. Install caffe: we use a modified version of the original caffe framework. Compile the source code in the ./caffe directory and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html).
 2. Download the 16-layer VGG network from https://gist.github.com/ksimonyan/211839e770f7b538e2d8, and put the caffemodel file under the ./model directory.
 3. Download imagenet-vgg-m-2048 from http://www.vlfeat.org/matconvnet/pretrained/, and put the file into ./networks
 4. Compile matconvnet in the sub-folders.
 5. Run the demo code demo_DRT.m. You can customize your own test sequences following this example.

The tracking results may be a little different on different machines. The suggested MATLAB and CUDA versions are MATLAB R2014B and CUDA 8.0. 

If you find our paper useful, please consider citing it.
@inproceedings{sun2018correlation,
    Author={Tiantian Wang and Lihe Zhang and Huchuan Lu and Chong Sun and Jinqing Qi},
    Title={Correlation Tracking via Joint Discrimination and Reliability Learning},
    Booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={489--497},
    Year={2018}
 }

