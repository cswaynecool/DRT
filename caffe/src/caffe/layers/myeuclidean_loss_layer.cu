#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void myEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   
   Dtype* target_real=Layer<Dtype>::neta_out_fft_real[0]->mutable_gpu_data();
   Dtype* target_imag=Layer<Dtype>::neta_out_fft_imag[0]->mutable_gpu_data();

   int count = Layer<Dtype>::neta_loss_fft_real[0]->count();
  
   Dtype* diff_real=Layer<Dtype>::neta_loss_fft_real[0]->mutable_gpu_data();  
   Dtype* diff_imag=Layer<Dtype>::neta_loss_fft_imag[0]->mutable_gpu_data();      
 caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      target_real,
      diff_real);

 caffe_gpu_sub(
      count,
      bottom[0]->gpu_data()+bottom[0]->offset(1),
      target_imag,
      diff_imag);

 
}

template <typename Dtype>
void myEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}

INSTANTIATE_LAYER_GPU_FUNCS(myEuclideanLossLayer);

}  // namespace caffe
