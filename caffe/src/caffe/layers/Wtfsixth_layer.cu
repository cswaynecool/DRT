#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void WtfsixthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
   
//   const float * data=  Layer<float>::input[0]->cpu_data(); 
 const Dtype* data=Layer<Dtype>::fifthlayer_data[0]->mutable_cpu_data();
//time_t start ,end ; double cost; 
//time(&start);
   int conv_in_channels=data[0]; int conv_in_height=data[1]; int conv_in_width=data[2]; int height_in=data[3];
   int width_in= data[4]; int kernel_h=data[5]; int kernel_w=data[6];int pad_h=data[7];int pad_w=data[8]; 
   int stride_h=data[9]; int stride_w=data[10];

// 接着我们列写正向传播函数
 //这里我们计算估计的输出
    int out_h = (conv_in_height - kernel_h+2*pad_h)/stride_h +1;
    int out_w = (conv_in_width - kernel_w+2*pad_w)/stride_w +1;
//printf("out_h out_w in the first Layer %d %d\n\n",out_h,out_w);
// Blob<Dtype> *col=new Blob<Dtype>(1,1,kernel_h*kernel_w*conv_in_channels,out_h*out_w);
      //    Dtype weight1[this->blobs_[0]->height()*this->blobs_[0]->width()*this->blobs_[0]->channels()];
printf("the first layer\n\n");
const Dtype* data1= bottom[0]->mutable_gpu_data();

      //    Dtype *col_buff=Layer<Dtype>::input[1]->mutable_gpu_data();
         Dtype *col_buff=Layer<Dtype>::sixthlayer_col_buff[0]->mutable_gpu_data(); 

im2col_gpu(data1, conv_in_channels, conv_in_height, conv_in_width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, col_buff, 1, 1);


  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data(); 

 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_in_channels*kernel_h*kernel_w,
        1,out_h*out_w,
        (Dtype)1., col_buff,weight,
        (Dtype)0., top_data); 
  }  
}

template <typename Dtype>
void WtfsixthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  LOG(INFO) << "start of convolutionlayer backward_gpu";
  //        sleep(10);
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin.";

          
          
        //  const Dtype * data=  Layer<Dtype>::input[0]->cpu_data();
const Dtype* data=Layer<Dtype>::fifthlayer_data[0]->mutable_cpu_data();
Dtype *bottom_data=bottom[0]->mutable_gpu_data();
const  int conv_in_channels_=data[0];const  int conv_in_height_=data[1];const int conv_in_width_=data[2];
const  int kernel_h_=data[5]; const int kernel_w_=data[6];const  int pad_h_=data[7]; const int pad_w_=data[8]; const int stride_h_=1;const int stride_w_=1; 

int out_h = (conv_in_height_ - kernel_h_+2*pad_h_)/stride_h_ +1;
    int out_w = (conv_in_width_ - kernel_w_+2*pad_w_)/stride_w_ +1;

  const Dtype* weight = this->blobs_[0]->gpu_data();
 Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
 
//  Dtype *col_buff=Layer<Dtype>::input[1]->mutable_gpu_data();

Dtype *col_buff=Layer<Dtype>::sixthlayer_col_buff[0]->mutable_gpu_data(); 



  im2col_gpu(bottom_data, conv_in_channels_,conv_in_height_, conv_in_width_,
             kernel_h_, kernel_w_, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
  // 我们首先要对top_diff进行处理 

const Dtype *top_diff=top[0]->gpu_diff();
//Blob<Dtype>* img_blob=new Blob<Dtype>(); img_blob->Reshape(1,conv_in_channels_,kernel_h_,kernel_w_);
//Dtype *data_im=img_blob->mutable_gpu_data();
    //      Dtype *data_im=Layer<Dtype>::input[3]->mutable_gpu_data();

 // col2im_gpu(top_diff, conv_in_channels_,
  //  kernel_h_, kernel_w_,kernel_h_/3 , kernel_w_,0,0,
  //  kernel_h_/3,
  //  kernel_w_/3,data_im);      
// 接着将data_im转化成k*1
     
        //  Dtype * col_buff1=Layer<Dtype>::input[2]->mutable_gpu_data();
   Dtype * col_buff1=Layer<Dtype>::sixthlayer_col_buff1[0]->mutable_gpu_data();

printf("size1 is %d %d %d %d %d\n\n\n",conv_in_channels_,conv_in_height_,conv_in_width_,kernel_h_,kernel_w_);

  im2col_gpu(top_diff, conv_in_channels_,kernel_h_, kernel_w_,
             kernel_h_, kernel_w_, 0, 0, 1, 1, col_buff1, 1,1); 
// 处理完毕
 
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,out_h*out_w,1, conv_in_channels_*kernel_h_*kernel_w_,
        (Dtype)1., col_buff,col_buff1,
        (Dtype)0., weight_diff);

  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfsixthLayer);

}  // namespace caffe
