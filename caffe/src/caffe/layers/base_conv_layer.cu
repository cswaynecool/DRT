#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void fill_mask_kernel(const int n, const unsigned int* mask_index,
    const int channel, const int height, const int width, 
    const int half_height, const int half_width, const int dim,
    unsigned int* mask) {
  CUDA_KERNEL_LOOP(index, n) {
   const int c1= index / dim;
   const int c2=c1/channel;
   const int c = index / dim;
   const int h = (index - c * dim) / width;
   const int w = index % width;
   const int h_ind = h / half_height;
   const int w_ind = w / half_width;
   mask[index] = mask_index[c2 * 25 + 5 * h_ind + w_ind];
  }
}

  
template <typename Dtype> 
void BaseConvolutionLayer<Dtype>::fill_mask_gpu() {
  const unsigned int* mask_index = this->mask_index_.gpu_data();
  unsigned int* mask = this->mask_.mutable_gpu_data();
  const int channel = this->mask_.channels();
  const int height = this->mask_.height();
  const int half_height = height / 5;
  const int width = this->mask_.width();
  const int half_width = width / 5;
  const int dim = height * width;
  const int count = this->mask_.count();
  fill_mask_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, mask_index, channel, height, width, half_height, half_width, dim, mask);
}
template void BaseConvolutionLayer<double>::fill_mask_gpu();
template void BaseConvolutionLayer<float>::fill_mask_gpu();

template <typename Dtype> 
__global__ void mul_kernel(const int n, Dtype* output, 
    const unsigned int* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    output[index] *= mask[index];
  }
}

template <typename Dtype> 
void BaseConvolutionLayer<Dtype>::forward_gpu_mask(Dtype* output, const unsigned int* mask) {
  const int count = this->mask_.count();
  mul_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, output, mask);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_mask(Dtype* output, const unsigned int* mask) {
  const int count = this->mask_.count();
  mul_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, output, mask);
}

template void BaseConvolutionLayer<double>::forward_gpu_mask(double* output, const unsigned int* mask);
template void BaseConvolutionLayer<float>::forward_gpu_mask(float* output, const unsigned int* mask);
template void BaseConvolutionLayer<double>::backward_gpu_mask(double* output, const unsigned int* mask);
template void BaseConvolutionLayer<float>::backward_gpu_mask(float* output, const unsigned int* mask);

}
