#include <vector>
#include <cfloat>


#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void myMaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void WtfseventhLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
          Dtype* data=Layer<Dtype>::dt_height[0]->mutable_cpu_data();
          Dtype* data1=Layer<Dtype>::dt_width[0]->mutable_cpu_data();
 int conv_in_height_=data[0]; 
 int conv_in_width_=data1[0]; 
 int conv_in_channels_=1;// int conv_in_height_=46; int conv_in_width_=46; 
 int kernel_h_=21; int kernel_w_=21; int pad_h_=10; int pad_w_=10;int stride_h_=1; int stride_w_=1;

printf("here no problem1\n");

Dtype* top_data=top[0]->mutable_gpu_data();
Dtype* tmp=Layer<Dtype>::seventhlayer_tmp[0]->mutable_gpu_data();
Dtype* x=Layer<Dtype>::seventhlayer_template_x[0]->mutable_gpu_data();
Dtype* x1=Layer<Dtype>::seventhlayer_template_x1[0]->mutable_gpu_data();
Dtype* y=Layer<Dtype>::seventhlayer_template_y[0]->mutable_gpu_data();
Dtype* y1=Layer<Dtype>::seventhlayer_template_y1[0]->mutable_gpu_data();

       //   printf("here no problem2\n");

 Dtype* bottom_data=bottom[0]->mutable_gpu_data();
 Dtype* col_buff=Layer<Dtype>::seventhlayer_col_buff[0]->mutable_gpu_data();

int* mask = this->dt_max_idx_.mutable_gpu_data();        

 for (int channel=0; channel<bottom[0]->channels();channel++)
//for (int channel=0; channel<1;channel++)
    { 
        int count=conv_in_height_*conv_in_width_;
        Dtype* weight=this->blobs_[0]->mutable_gpu_data()+4*channel;
            im2col_gpu(bottom_data+conv_in_height_*conv_in_width_*channel, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff, 1, 1); 
      
        caffe_gpu_dtpooling(kernel_h_*kernel_w_*1*conv_in_height_*conv_in_width_, col_buff, weight, x,
        x1, y, y1,conv_in_height_, conv_in_width_, tmp); 

    // NOLINT_NEXT_LINE(whitespace/operators)
        myMaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
           count, tmp, tmp+conv_in_height_*conv_in_width_, 0, top_data+conv_in_height_*conv_in_width_*channel, mask+conv_in_height_*conv_in_width_*channel);

    for (int i = 2; i < 441; ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
        myMaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data+conv_in_height_*conv_in_width_*channel, tmp+conv_in_height_*conv_in_width_*i, i-1, top_data+conv_in_height_*conv_in_width_*channel, mask+conv_in_height_*conv_in_width_*channel);
        }
    }
}
    
 
template <typename Dtype>
__global__ void myMaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask,Dtype* x, Dtype* x1, Dtype* y, Dtype* y1,int conv_in_height_, int conv_in_width_, Dtype* tmp1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  int mask_index=mask[index];
  tmp1[index+conv_in_height_*conv_in_width_*0]=x[mask_index];
  tmp1[index+conv_in_height_*conv_in_width_*1]=x1[mask_index];
  tmp1[index+conv_in_height_*conv_in_width_*2]=y[mask_index];
  tmp1[index+conv_in_height_*conv_in_width_*3]=y1[mask_index];
  }   
}  

template <typename Dtype>
void WtfseventhLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
int conv_in_channels_=1; 
  Dtype* data=Layer<Dtype>::dt_height[0]->mutable_cpu_data();
  Dtype* data1=Layer<Dtype>::dt_width[0]->mutable_cpu_data();
 int conv_in_height_=data[0]; 
 int conv_in_width_=data1[0]; 
 int kernel_h_=21; int kernel_w_=21; int pad_h_=10; int pad_w_=10;int stride_h_=1; int stride_w_=1;

Dtype* top_data=top[0]->mutable_gpu_data();
Dtype* tmp=Layer<Dtype>::seventhlayer_tmp[0]->mutable_gpu_data();
Dtype* x=Layer<Dtype>::seventhlayer_template_x[0]->mutable_gpu_data();
Dtype* x1=Layer<Dtype>::seventhlayer_template_x1[0]->mutable_gpu_data();
Dtype* y=Layer<Dtype>::seventhlayer_template_y[0]->mutable_gpu_data();
Dtype* y1=Layer<Dtype>::seventhlayer_template_y1[0]->mutable_gpu_data();

Dtype* top_diff=top[0]->mutable_gpu_diff();

Dtype* weight_diff=this->blobs_[0]->mutable_gpu_diff();
int count=conv_in_height_*conv_in_width_;
 Dtype* bottom_data=bottom[0]->mutable_gpu_data();
 Dtype* col_buff=Layer<Dtype>::seventhlayer_col_buff[0]->mutable_gpu_data();

int* mask = this->dt_max_idx_.mutable_gpu_data();         
Dtype* tmp1=Layer<Dtype>::seventhlayer_tmp1[0]->mutable_gpu_data();

//int* mask_cpu=this->dt_max_idx_.mutable_cpu_data();         
//for(int i=0; i<this->dt_max_idx_.count();i++)
//{
//printf("the value is %d \n",mask_cpu[i]);
//}

//caffe_copy(Layer<Dtype>::mask_out[0]->count(),(Dtype*)mask, Layer<Dtype>::mask_out[0]->mutable_gpu_data());

    for (int channel=0; channel<bottom[0]->channels();channel++)
//for (int channel=0; channel<1;channel++)
    {

        myMaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                 count, top_diff, channel, mask+conv_in_height_*conv_in_width_*channel,x,x1,y,y1,conv_in_height_,conv_in_width_, tmp1);

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
         4,conv_in_height_*conv_in_width_,
        (Dtype)1., top_diff+conv_in_height_*conv_in_width_*channel, tmp1,
        (Dtype)0., weight_diff+4*channel); 


    }
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfseventhLayer);

}  // namespace caffe
