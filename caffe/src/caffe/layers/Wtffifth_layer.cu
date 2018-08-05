#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void WtffifthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
 const Dtype* weight = this->blobs_[0]->gpu_data();
      //  printf("this values is %d-> %d-> %d-> %d\n\n",this->blobs_[0]->num(), this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width());
         // sleep(10);
  const Dtype * data=  Layer<Dtype>::fifthlayer_data[0]->cpu_data(); 
  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1;

    const Dtype* bottom_data = bottom[0]->gpu_data();
  //  Dtype *data_im=Layer<Dtype>::input[6]->mutable_gpu_data();
    Dtype *data_im=Layer<Dtype>::fifthlayer_data_im[0]->mutable_gpu_data(); 

   col2im_gpu(bottom_data, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_, kernel_w_,0,0,
    1,1,data_im);      

    Dtype* top_data = top[0]->mutable_gpu_data();
//我们将输入的image分为9个部分来做
// 接着我们再将得到的结果重新进行分块
    //    Dtype* template1=Layer<Dtype>::input[4]->mutable_gpu_data();
          Dtype* template1=Layer<Dtype>::fifthlayer_template1[0]->mutable_gpu_data();
im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);


// 注意 如果改成5个patch 必须改下面的语句～～～～～～～～～～～～～～～～～～～～
    int out_h = (Layer<Dtype>::fifthlayer_tmp[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::fifthlayer_tmp[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;
 

  //  Dtype * col_buff=Layer<Dtype>::input[8]->mutable_gpu_data();
     Dtype * col_buff=Layer<Dtype>::fifthlayer_col_buff[0]->mutable_gpu_data(); 

   printf("the out_h out_w are %d %d\n\n",out_h,out_w);


//Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data(); 
   Dtype *  data1= Layer<Dtype>::fifthlayer_tmp[0]->mutable_gpu_data(); 


     for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
        { 
            int layer_index=3*ii+jj;  
            im2col_gpu(data1+Layer<Dtype>::fifthlayer_tmp[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::fifthlayer_tmp[0]->height() , Layer<Dtype>::fifthlayer_tmp[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
        // 接着计算该数据与template1的内积，所得的结果应为9 out_h*out_w
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,9 , out_h*out_w, conv_in_channels_*kernel_h_/3*kernel_w_/3,
            (Dtype)1., template1 , col_buff,
            (Dtype)0, top_data+layer_index*out_h*out_w*9);
        }
       
}    
 
    
  

template <typename Dtype>
void WtffifthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
//Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data(); 
 Dtype *  data1= Layer<Dtype>::fifthlayer_tmp[0]->mutable_gpu_data();  
          
          const Dtype* weight = this->blobs_[0]->gpu_data();
 Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff(); 
 const Dtype* top_diff=top[0]->gpu_diff();
Dtype * bottom_diff=bottom[0]->mutable_gpu_diff();
const Dtype* bottom_data = bottom[0]->gpu_data();
// 一些会用到的参数
// const Dtype * data=  Layer<Dtype>::input[0]->cpu_data();
   const Dtype * data=Layer<Dtype>::fifthlayer_data[0]->cpu_data(); 

  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1; 
   
    Dtype* top_data = top[0]->mutable_gpu_data();

      //    Dtype *template1=Layer<Dtype>::input[9]->mutable_gpu_data();
     Dtype* template1=Layer<Dtype>::fifthlayer_template1[0]->mutable_gpu_data();

//首先将上一层的特征转换成图片
      //    Dtype* data_im=Layer<Dtype>::input1[4]->mutable_gpu_data();
     Dtype *data_im=Layer<Dtype>::fifthlayer_data_im[0]->mutable_gpu_data(); 


col2im_gpu(bottom_data, conv_in_channels_,
    conv_in_height_, conv_in_width_, kernel_h_, kernel_w_,
    0, 0,  stride_h_,
    stride_w_,data_im);


im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);

// 定义一临时存放9个patch的变量
    int out_h = (Layer<Dtype>::fifthlayer_tmp[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::fifthlayer_tmp[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;


// Dtype *col_buff=Layer<Dtype>::input[11]->mutable_gpu_data();
          Dtype* col_buff=Layer<Dtype>::fifthlayer_col_buff[0]->mutable_gpu_data();

 // Dtype* tmp4=Layer<Dtype>::input[13]->mutable_gpu_data();
   Dtype* tmp4=Layer<Dtype>::fifthlayer_tmp4[0]->mutable_gpu_data(); 

// 接着我们利用top残差top_diff来计算当前layer的梯度，显然，上一层的残差维度为out_h*out_w
 for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
          {//
             int layer_index=3*ii+jj;  

             im2col_gpu(data1+Layer<Dtype>::fifthlayer_tmp[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::fifthlayer_tmp[0]->height() , Layer<Dtype>::fifthlayer_tmp[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
        // 接着我们计算bottom_diff
             if(ii==0&&jj==0) 
             {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,conv_in_channels_*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w, 
                (Dtype)1.,col_buff,  top_diff+layer_index*out_h*out_w*9,
                (Dtype)0,tmp4 ); 
             }
             else
             {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,conv_in_channels_*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w, 
                (Dtype)1.,col_buff,  top_diff+layer_index*out_h*out_w*9,
                (Dtype)1,tmp4 );  
             }
          }

// 接着，我们对tmp4进行处理，以期得到bottom_diff

Dtype* tmp5=Layer<Dtype>::fifthlayer_tmp5[0]->mutable_gpu_data();

col2im_gpu(tmp4, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_/3 , kernel_w_/3,0,0,
    kernel_h_/3, kernel_w_/3,tmp5);      

im2col_gpu(tmp5, conv_in_channels_,kernel_h_, kernel_w_,
             kernel_h_, kernel_w_, pad_h_, pad_w_, 1, 1, bottom_diff, 1,1);
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtffifthLayer);

}  // namespace caffe
