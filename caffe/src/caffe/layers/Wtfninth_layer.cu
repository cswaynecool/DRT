#include <vector>
#include <cfloat>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void myMaxForward1(const int nthreads, const Dtype* bottom_data_a,
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
void WtfninthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
      //  printf("this values is %d-> %d-> %d-> %d\n\n",this->blobs_[0]->num(), this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width());
         // sleep(10);
  const Dtype * data=  Layer<Dtype>::input[0]->cpu_data(); 
  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1;
   const Dtype * parameter=Layer<Dtype>::input1[8]->cpu_data();
   Dtype lambda1=parameter[0]; Dtype lambda2=parameter[1];
//printf("the second layer\n\n");
  Dtype* x=Layer<Dtype>::seventhlayer_template_x[0]->mutable_gpu_data();
  Dtype* y=Layer<Dtype>::seventhlayer_template_y[0]->mutable_gpu_data();
  Dtype* x1=Layer<Dtype>::seventhlayer_template_x1[0]->mutable_gpu_data();
  Dtype* y1=Layer<Dtype>::seventhlayer_template_y1[0]->mutable_gpu_data();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype *data_im=Layer<Dtype>::input[6]->mutable_gpu_data();
     

   col2im_gpu(bottom_data, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_, kernel_w_,0,0,
    1,1,data_im);      

    Dtype* top_data = top[0]->mutable_gpu_data();
//我们将输入的image分为9个部分来做
// 接着我们再将得到的结果重新进行分块
        Dtype* template1=Layer<Dtype>::input[4]->mutable_gpu_data();
im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);


// 定义一临时存放9个patch的变量
    int out_h = (Layer<Dtype>::ninthlayer_tmp[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::ninthlayer_tmp[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;

  //  Blob<Dtype>* tmp1_blob=new Blob<Dtype>(); tmp1_blob->Reshape(1,1,9,out_h*out_w);
   //       Dtype *tmp1=tmp1_blob->mutable_gpu_data();
   
  //  Dtype* tmp1=Layer<Dtype>::input[7]->mutable_gpu_data();  
      Dtype* tmp1=Layer<Dtype>::ninthlayer_tmp1[0]->mutable_gpu_data();


  //  Dtype * col_buff=Layer<Dtype>::input[8]->mutable_gpu_data();
  Dtype* col_buff=Layer<Dtype>::ninthlayer_col_buff[0]->mutable_gpu_data();

//Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data();
  Dtype* data1=Layer<Dtype>::ninthlayer_tmp[0]->mutable_gpu_data(); 
    
 Dtype* template_tmp=Layer<Dtype>::ninthlayer_template_tmp[0]->mutable_gpu_data();

 Dtype* col_buff1=Layer<Dtype>::ninthlayer_col_buff1[0]->mutable_gpu_data();

 Dtype* tmp2=Layer<Dtype>::ninthlayer_tmp2[0]->mutable_gpu_data();

int* mask=this->ninthlayer_max_idx_.mutable_gpu_data();

     for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
        { 
            int layer_index=3*ii+jj;  
            im2col_gpu(data1+Layer<Dtype>::ninthlayer_tmp[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::ninthlayer_tmp[0]->height() , Layer<Dtype>::ninthlayer_tmp[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
        // 从template1中选择一列 
            caffe_gpu_obtain_col(Layer<Dtype>::input[4]->count(), template1, Layer<Dtype>::input[4]->height(), Layer<Dtype>::input[4]->width(), layer_index, template_tmp); 
            

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,1 , out_h*out_w, conv_in_channels_*kernel_h_/3*kernel_w_/3,
            (Dtype)1., template_tmp , col_buff,
            (Dtype)0, tmp1+Layer<Dtype>::ninthlayer_tmp1[0]->offset(layer_index));
         
         //接着我们将tmp1按列展开
          im2col_gpu(tmp1+Layer<Dtype>::ninthlayer_tmp1[0]->offset(layer_index), 1, out_h, out_w,
             11, 11, 5, 5, 1, 1, col_buff1, 1,1); 

        //  caffe_gpu_refinenet(const int N, const double* col_buff, double* weight,double* x, double* x1, double* y, double* y1, int width, double* output) 
            caffe_gpu_refinenet(Layer<Dtype>::ninthlayer_col_buff1[0]->count(), col_buff1, weight+layer_index*4, x, x1, y, y1, out_h*out_w, tmp2); 
         int count=out_h*out_w; 
       // 接着开始最大化操作
            myMaxForward1<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, tmp2, tmp2+out_h*out_w, 0, top_data+out_h*out_w*layer_index, mask+out_h*out_w*layer_index);

           for (int i = 2; i < 121; ++i) 
            {
              myMaxForward1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
               count, top_data+out_h*out_w*layer_index, tmp2+out_h*out_w*i, i-1, top_data+out_h*out_w*layer_index, mask+out_h*out_w*layer_index);
            } 
   

        } 

}

template <typename Dtype>
__global__ void myMaxBackward1(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask,Dtype* x, Dtype* x1, Dtype* y, Dtype* y1,int out_h, int out_w, Dtype* tmp1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  int mask_index=mask[index];
  tmp1[index+out_h*out_w*0]=x[mask_index];
  tmp1[index+out_h*out_w*1]=x1[mask_index];
  tmp1[index+out_h*out_w*2]=y[mask_index];
  tmp1[index+out_h*out_w*3]=y1[mask_index];
  }   
}  


template <typename Dtype>
void WtfninthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  LOG(INFO) << "start of convolutionlayer backward_gpu";
  //        sleep(10);
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin.";
 //  Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data(); 
   Dtype* data1=Layer<Dtype>::ninthlayer_tmp[0]->mutable_gpu_data(); 

  const Dtype* weight = this->blobs_[0]->gpu_data();
 Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff(); 
 const Dtype* top_diff=top[0]->gpu_diff();
Dtype * bottom_diff=bottom[0]->mutable_gpu_diff();
const Dtype* bottom_data = bottom[0]->gpu_data();
// 一些会用到的参数
 const Dtype * data=  Layer<Dtype>::input[0]->cpu_data(); 
  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1; 
  
  Dtype* x=Layer<Dtype>::seventhlayer_template_x[0]->mutable_gpu_data();
  Dtype* y=Layer<Dtype>::seventhlayer_template_y[0]->mutable_gpu_data();
  Dtype* x1=Layer<Dtype>::seventhlayer_template_x1[0]->mutable_gpu_data();
  Dtype* y1=Layer<Dtype>::seventhlayer_template_y1[0]->mutable_gpu_data();

 int* mask=this->ninthlayer_max_idx_.mutable_gpu_data();

    Dtype* top_data = top[0]->mutable_gpu_data();

          Dtype *template1=Layer<Dtype>::input[9]->mutable_gpu_data();

//首先将上一层的特征转换成图片
          Dtype* data_im=Layer<Dtype>::input1[4]->mutable_gpu_data();
col2im_gpu(bottom_data, conv_in_channels_,
    conv_in_height_, conv_in_width_, kernel_h_, kernel_w_,
    0, 0,  stride_h_,
    stride_w_,data_im);


im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);

// 定义一临时存放9个patch的变量
    int out_h = (Layer<Dtype>::ninthlayer_tmp[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::ninthlayer_tmp[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;

    Dtype *col_buff=Layer<Dtype>::ninthlayer_col_buff[0]->mutable_gpu_data();

    Dtype* template_tmp=Layer<Dtype>::ninthlayer_template_tmp[0]->mutable_gpu_data();
   
    Dtype* tmp3=Layer<Dtype>::ninthlayer_tmp3[0]->mutable_gpu_data();

    Dtype* tmp4=Layer<Dtype>::ninthlayer_tmp4[0]->mutable_gpu_data(); 
// 接着我们利用top残差top_diff来计算当前layer的梯度，显然，上一层的残差维度为out_h*out_w
 for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
          {//
             int count=out_h*out_w;
             int layer_index=3*ii+jj;
            im2col_gpu(data1+Layer<Dtype>::ninthlayer_tmp[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::ninthlayer_tmp[0]->height() , Layer<Dtype>::ninthlayer_tmp[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
        

        // 接下来我们计算梯度 
           myMaxBackward1<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                 count, top_diff, layer_index, mask+out_h*out_w*layer_index,x,x1,y,y1,out_h,out_w, tmp3); 
             caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
             4,out_h*out_w,(Dtype)1., top_diff+out_h*out_w*layer_index, tmp3,
             (Dtype)0., weight_diff+4*layer_index); 
           //首先拿到当前layer index下的template column  
           //然后我们尝试对改
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
             conv_in_channels_*kernel_h_*kernel_w_/9,out_h*out_w,(Dtype)1., top_diff+out_h*out_w*layer_index, col_buff,
             (Dtype)0.,template_tmp);

             caffe_gpu_placecol(kernel_h_*kernel_w_/9*conv_in_channels_, template_tmp, layer_index,conv_in_channels_*kernel_h_*kernel_w_/9, 9, template1);

          }

    col2im_gpu(template1, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_/3 , kernel_w_/3,0,0,
    kernel_h_/3, kernel_w_/3,bottom_diff);      

  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfninthLayer);

}  // namespace caffe
