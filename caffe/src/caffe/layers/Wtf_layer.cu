#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>  
#include <cuda_runtime.h>
#include <cufft.h>
#include "common/inc/helper_functions.h"
#include "common/inc/helper_cuda.h"
typedef float2 Complex;
#define SIGNAL_SIZE        50
#define FILTER_KERNEL_SIZE 11

namespace caffe {

void fft2(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

__global__ void copy_memory_to_blob(const int n, float2* mem1, float* tmp1, float* tmp2) {
  CUDA_KERNEL_LOOP(index, n) {
  }
}

__global__ void copy_memory_from_blob(const int n, float2* mem1, float* tmp1, float* tmp2) {
  CUDA_KERNEL_LOOP(index, n) {
    
  }
}   

template <typename Dtype>
__global__ void set_zeros(const int n, Dtype* in_out) {
  CUDA_KERNEL_LOOP(index, n) {
  in_out[index]=0;
  }
}

__global__ void scale_out_real(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}

template <typename Dtype>   
__global__ void add_mask(const int n, int num_per_channel, Dtype* mask, float* input, float * output) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
   output[index]=input[index]*mask[current_index];

  }
}

template <typename Dtype>
__global__ void ifftshift(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel1;
   int current_index=index%num_per_channel1;

   if(L_mask[current_index]>0) 
   {int ori_index=L_mask[current_index]-1+channel_index*num_per_channel1;
    output[index].x=input_real[ori_index];
    output[index].y=input_imag[ori_index];
   }
   else
   { int ori_index=-L_mask[current_index]-1+channel_index*num_per_channel1;
     output[index].x=input_real[ori_index];
     output[index].y=-input_imag[ori_index]; 
   }
  }
}

template <typename Dtype>
__global__ void fftshift(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel1;
   int current_index=index%num_per_channel1;

   if(L_mask[current_index]>-0.5)
    {
      int ori_index=L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=input[ori_index].y; 
    }
    else
    {
      int ori_index=-L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=-input[ori_index].y;  
    }
  }
}   

template <typename Dtype>
__global__ void obtain_output(const int n,int number_per_sample1, int number_per_sample2,Dtype* L_mask, Dtype* real1, Dtype* real2, Dtype* real3, Dtype* real4, Dtype* real5,Dtype* imag1,Dtype* imag2, Dtype* imag3,Dtype* imag4,Dtype* imag5,Dtype* y_real, Dtype* y_imag) {
  CUDA_KERNEL_LOOP(index, n) {
    int sample_index1=index/number_per_sample1;
    int index1=index%number_per_sample1;
    int index2=number_per_sample2*sample_index1+L_mask[index1]-1;
   if(L_mask[index1]==0)
      {
        y_real[index]=real1[index]+real2[index]+real4[index]+real5[index];
        y_imag[index]=imag1[index]+imag2[index]+imag4[index]+imag5[index]; 
      }
   else
      { 
        y_real[index]=real1[index]+real2[index]+real3[index2]+real4[index]+real5[index];
        y_imag[index]=imag1[index]+imag2[index]+imag3[index2]+imag4[index]+imag5[index]; 

      }
  }
}

template <typename Dtype>
__global__ void obtain_freq(const int n, float2* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
  output[index]=input[index].x;
  output[index+n]=input[index].y; 
  }
}   


template <typename Dtype>   
 __global__ void pad_filter(const int n,Dtype* pad_mask, int pad_h, int pad_w, int num_per_channel1, int num_per_channel2, int filter_h, int filter_w, int height, int width, int padded_height, int padded_width, Dtype* h_real_in, Dtype* h_imag_in , Dtype* h_real_out, Dtype* h_imag_out) {
 CUDA_KERNEL_LOOP(index, n) {
   int current_index=index%num_per_channel1;
   int channel_index=index/num_per_channel1;
   int index_ori=pad_mask[current_index]+channel_index*num_per_channel2;
   h_real_out[index]=h_real_in[0];
   h_imag_out[index]=h_imag_in[0]; 
  }
}

template <typename Dtype>   
__global__ void get_col(const int n, Dtype* col_mask, Dtype* h_real_in, Dtype* h_imag_in, Dtype* h_real_col, Dtype* h_imag_col) {
  CUDA_KERNEL_LOOP(index, n) {
  int index_ori=col_mask[index];
  h_real_col[index]=h_real_in[index_ori];
  h_imag_col[index]=h_imag_in[index_ori];
  

  }
}

template <typename Dtype>   
__global__ void get_freq(const int n, float2* freq, Dtype* top_data_real, Dtype* top_data_imag) {
  CUDA_KERNEL_LOOP(index, n) {
  top_data_real[index]=freq[index].x;
  top_data_imag[index]=freq[index].y; 

  }
}

template <typename Dtype>   
__global__ void set_freq(const int n, float2* freq, Dtype* input_data) {
  CUDA_KERNEL_LOOP(index, n) {
  freq[index].x=input_data[index];
  freq[index].y=input_data[index+n];

  }
}

template <typename Dtype>   
__global__ void laplace_add(const int n, Dtype* input1, Dtype* input2, Dtype* output1, Dtype* output2,Dtype factor) {
  CUDA_KERNEL_LOOP(index, n) {
  output1[index]=output1[index]+factor*input1[index];
  output2[index]=output2[index]+factor*input2[index];
  }
}


template <typename Dtype>
__global__ void my_weight_sample_kernel(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weight_real, Dtype* weight_imag, Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
 CUDA_KERNEL_LOOP(index, n) { 
    int channel_num=number_per_sample/number_per_channel;
    int sample_index=index/number_per_channel;
    int position_index=index%number_per_channel;
    for(int i=0;i<channel_num;i++)
    {int hf_base_position=position_index+i*number_per_channel;
     weighted_sample_real[index]= weighted_sample_real[index]+weight_real[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index];
    weighted_sample_imag[index]= weighted_sample_imag[index]-weight_real[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index];
    } 

  }
}

template <typename Dtype>
__global__ void weight_sample_kernel_second(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weighted_sample_real, Dtype* weighted_sample_imag, Dtype* KK_real,Dtype* KK_imag,Dtype* sample_weight, int number_per_sample,int number_per_channel, int sample_num) {
  CUDA_KERNEL_LOOP(index, n) {
 
  int position_index=index%number_per_channel;

    for(int i=0; i<sample_num;i++)
     {
        int weighted_sample_index=position_index+i*number_per_channel;
        int index1=index+i*number_per_sample;
        KK_real[index]=KK_real[index]+sample_weight[i]*(weighted_sample_real[weighted_sample_index]*sample_real[index1]-weighted_sample_imag[weighted_sample_index]*sample_imag[index1]);
        KK_imag[index]=KK_imag[index]+sample_weight[i]*(weighted_sample_real[weighted_sample_index]*sample_imag[index1]+weighted_sample_imag[weighted_sample_index]*sample_real[index1]);
      }



  }
}

template <typename Dtype>
__global__ void fuse_result(const int n, Dtype* input,Dtype* output, int channels, int num_per_channel2,int number_per_sample1 ) {
  CUDA_KERNEL_LOOP(index, n) {
   for(int frag_id=0;frag_id<10;frag_id++)
    {  int position_index=index+number_per_sample1*frag_id;
         if(frag_id<9)
        {
          output[index]=output[index]+9*input[position_index];
        }
        else
        {
          output[index]=output[index]-input[position_index]; 
        }
    }

  }
}

template <typename Dtype>
__global__ void add_different_layers(const int n,int num_per_channel1, int num_per_channel2, Dtype* L_mask, Dtype* real,Dtype* imag, Dtype* sh_real, Dtype* sh_imag) {
  CUDA_KERNEL_LOOP(index, n) {
    int channel_index=index/num_per_channel1;
    int index1=index%num_per_channel1;
    int index2=num_per_channel2*channel_index+L_mask[index1]-1;
   if(L_mask[index1]==0)
      {
         sh_real[index]=sh_real[index]; 
         sh_imag[index]=sh_imag[index];
      }
   else
      { 
         sh_real[index]=sh_real[index]+real[index2];
         sh_imag[index]=sh_imag[index]+imag[index2];
      }
  }
}

template <typename Dtype>
__global__ void crop_sample(const int n,int num_per_channel1, int num_per_channel2, Dtype* L_mask1, Dtype* sh_real, Dtype* sh_imag, Dtype* output_real, Dtype* output_imag) {
  CUDA_KERNEL_LOOP(index, n) {
      int position_index=index%num_per_channel1;
      int channel_index=index/num_per_channel1;
      int index1=(L_mask1[position_index]-1)+num_per_channel2*channel_index;
      output_real[index]=sh_real[index1];
      output_imag[index]=sh_imag[index1];
  }
}


template <typename Dtype>
void WtfLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];

    int count1; int count2;  int count3;  int number_per_sample; float scale_factor; int col_num; int row_num; int num_per_channel1;
    int num_per_channel2;

    Dtype* frame_id_cpu=Layer<Dtype>::frame[0]->mutable_cpu_data();
    int frame_id=frame_id_cpu[0];

    Dtype* sample_weight=Layer<Dtype>::sample_weight[0]->mutable_gpu_data();

    int sample_num=Layer<Dtype>::sample_weight[0]->width();

    Dtype* index_cpu=Layer<Dtype>::index[0]->mutable_cpu_data();
    Dtype* index_cpu1=Layer<Dtype>::index1[0]->mutable_cpu_data();
    int index[feature_num];
    int index1[feature_num];
    for(int i=0;i<feature_num;i++)
    {
       index[i]=index_cpu[i];
       index1[i]=index_cpu1[i]; 
    }


    Dtype* ifftshift_mask;Dtype* fftshift_mask; Dtype* weighted_sample_real;Dtype* weighted_sample_imag;
    Dtype* sample_real; Dtype* sample_imag;
    Dtype* KK_real;Dtype* KK_imag;
    Dtype* tmp_real1;Dtype* tmp_imag1;
    Dtype* hf_real;
    Dtype* hf_imag; Dtype* laplace_real; Dtype* laplace_imag; Dtype* mask;
  
   for(int blob_id=0;blob_id<feature_num; blob_id++)
    {     
      if(blob_id!=2)
      {  
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
          ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor); 

         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);
 
         for(int frag_id=0; frag_id<10; frag_id++)  
          {
             if(frag_id<9)
             { 
               mask=Layer<Dtype>::patch_mask[0]->mutable_gpu_data()+row_num*col_num*frag_id;
               add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_total1+frag_id*count2+index1[blob_id]); 
             }
             else
              {
                mask=Layer<Dtype>::binary_mask[0]->mutable_gpu_data();
                add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_total1+frag_id*count2+index1[blob_id]);  
              }
          }

      }
    }  

fft2(this->forward_plan_total[0],this->d_in_total1,this->d_freq_total1);
    
for(int blob_id=0;blob_id<feature_num; blob_id++)
{

    if(blob_id!=2)
    { 
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num; 

       for(int frag_id=0;frag_id<10;frag_id++)
        {
           
            fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq_total1+frag_id*count1+index[blob_id],Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());
             
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 


            if(frag_id==1)
            {  count3=count3/sample_num;}
            

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample,
                                                                                          num_per_channel1); 
  
            set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,KK_real);
            set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,KK_imag);

            if(frag_id==1)
            {
              weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, 1); 
            }
            else
            {
            weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, sample_num);
            }
            ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, KK_real , KK_imag, this->d_freq_total1+frag_id*count1+index[blob_id],row_num, col_num,num_per_channel1);

        }

    } 

}

Dtype* total_num_cpu=Layer<Dtype>::total_num[0]->mutable_cpu_data();
Dtype* total_num_cpu1=Layer<Dtype>::total_num1[0]->mutable_cpu_data();
int count4=total_num_cpu[0]*10;
int count5=total_num_cpu1[0]*10;
ifft2(this->inverse_plan_total[0],this->d_freq_total1,this->d_in_total1);
scale_out_real<<<CAFFE_GET_BLOCKS(count4), CAFFE_CUDA_NUM_THREADS>>>(count4,this->d_in_total1,scale_factor);

for(int blob_id=0; blob_id<feature_num;blob_id++)
{ 
  if(blob_id!=2)
  {
    count2=this->blobs_[blob_id]->channels()*row_num*col_num; 
   for(int frag_id=0;frag_id<10;frag_id++)
    {
        if(frag_id<9)
        {  
           mask=Layer<Dtype>::patch_mask[0]->mutable_gpu_data()+row_num*col_num*frag_id;
           add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in_total1+frag_id*count2+index1[blob_id], this->d_in_total1+frag_id*count2+index1[blob_id]); 
        }
        else
        {
           mask=Layer<Dtype>::binary_mask[0]->mutable_gpu_data();
           add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in_total1+frag_id*count2+index1[blob_id], this->d_in_total1+frag_id*count2+index1[blob_id]);  

        }
    }
   }
}

for(int blob_id=0;blob_id<feature_num;blob_id++)
{
   if(blob_id!=2)
   {
    int count6=this->blobs_[blob_id]->channels()*num_per_channel2;
    int number_per_sample1=this->blobs_[blob_id]->channels()*num_per_channel2;
    set_zeros<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,this->d_in_sub_total1+index1[blob_id]/10);
    fuse_result<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,this->d_in_total1+index1[blob_id],this->d_in_sub_total1+index1[blob_id]/10,this->blobs_[blob_id]->channels(),   num_per_channel2,number_per_sample1);
   }
}

fft2(this->forward_plan_sub_total[0],this->d_in_sub_total1,this->d_freq_sub_total1);

for(int blob_id=0;blob_id<feature_num;blob_id++)
{ 
  if(blob_id!=2)
   { 
    laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
   laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data(); 
   int  count6=this->blobs_[blob_id]->channels()*num_per_channel1;
   fftshift<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,num_per_channel1,fftshift_mask,this->d_freq_sub_total1+index[blob_id]/10,laplace_real,laplace_imag);
   }
}

//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
 for(int blob_id=2;blob_id<3; blob_id++)
    {     
       if(blob_id==2) 
      {  
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
          ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor); 

         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);

         for(int frag_id=0; frag_id<10; frag_id++)  
          {
             if(frag_id<9)
             { 
               mask=Layer<Dtype>::patch_mask[1]->mutable_gpu_data()+row_num*col_num*frag_id;
               add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_total2+frag_id*count2+index1[blob_id]); 
             }
             else
              {
                mask=Layer<Dtype>::binary_mask[1]->mutable_gpu_data();
                add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_total2+frag_id*count2+index1[blob_id]);  
              }
          }

      }
    }  

fft2(this->forward_plan_total[1],this->d_in_total2,this->d_freq_total2);
    
for(int blob_id=2;blob_id<3; blob_id++)
{

    if(blob_id==2)
    { 
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num; 

       for(int frag_id=0;frag_id<10;frag_id++)
        {
           
            fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq_total2+frag_id*count1+index[blob_id],Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());
             
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 

            if(frag_id==1)
            {
                count3=count3/sample_num;
            }

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample,
                                                                                          num_per_channel1); 
  
            set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,KK_real);
            set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,KK_imag);

            if(frag_id==1)
            {
            weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, 1); 
            }
            else
            {
               weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, sample_num);  
            }
            ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, KK_real , KK_imag, this->d_freq_total2+frag_id*count1+index[blob_id],row_num, col_num,num_per_channel1);

        }

    } 

}

total_num_cpu=Layer<Dtype>::total_num2[0]->mutable_cpu_data();
total_num_cpu1=Layer<Dtype>::total_num3[0]->mutable_cpu_data();
count4=total_num_cpu[0]*10;
count5=total_num_cpu1[0]*10;
ifft2(this->inverse_plan_total[1],this->d_freq_total2,this->d_in_total2);
scale_out_real<<<CAFFE_GET_BLOCKS(count4), CAFFE_CUDA_NUM_THREADS>>>(count4,this->d_in_total2,scale_factor);

for(int blob_id=2; blob_id<3;blob_id++)
{ 
  if(blob_id==2)
  {
    count2=this->blobs_[blob_id]->channels()*row_num*col_num; 
   for(int frag_id=0;frag_id<10;frag_id++)
    {
        if(frag_id<9)
        {  
           mask=Layer<Dtype>::patch_mask[1]->mutable_gpu_data()+row_num*col_num*frag_id;
           add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in_total2+frag_id*count2+index1[blob_id], this->d_in_total2+frag_id*count2+index1[blob_id]); 
        }
        else
        {
           mask=Layer<Dtype>::binary_mask[1]->mutable_gpu_data();
           add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in_total2+frag_id*count2+index1[blob_id], this->d_in_total2+frag_id*count2+index1[blob_id]);  

        }
    }
   }
}

for(int blob_id=2;blob_id<3;blob_id++)
{
   if(blob_id==2)
   {
    int count6=this->blobs_[blob_id]->channels()*num_per_channel2;
    int number_per_sample1=this->blobs_[blob_id]->channels()*num_per_channel2;
    set_zeros<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,this->d_in_sub_total2+index1[blob_id]/10);
    fuse_result<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,this->d_in_total2+index1[blob_id],this->d_in_sub_total2+index1[blob_id]/10,this->blobs_[blob_id]->channels(),   num_per_channel2,number_per_sample1);
   }
}
fft2(this->forward_plan_sub_total[1],this->d_in_sub_total2,this->d_freq_sub_total2);


for(int blob_id=2;blob_id<3;blob_id++)
{ 
  if(blob_id==2)
   { 
    laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
   laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data(); 
   int  count6=this->blobs_[blob_id]->channels()*num_per_channel1;
   fftshift<<<CAFFE_GET_BLOCKS(count6), CAFFE_CUDA_NUM_THREADS>>>(count6,num_per_channel1,fftshift_mask,this->d_freq_sub_total2+index[blob_id]/10,laplace_real,laplace_imag);
   }
}




  for(int blob_id=0;blob_id<feature_num;blob_id++)
  {
    caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data());
    caffe_copy(this->blobs_[blob_id]->count()/2,Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data(),this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->count()/2);
  }
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
//××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
   for(int blob_id=0;blob_id<feature_num; blob_id++)
    {     
      if(blob_id!=2)
      {  
         ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
         fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data(); 

         weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
         weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

         sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
         sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

         KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
         KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

         tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
         tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
         hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
         hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

        laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
        laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();
         col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
         num_per_channel2=row_num*col_num; 
          
          count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
          count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
          ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor); 

         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
         set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);

             mask=Layer<Dtype>::binary_mask[0]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);
             fft2(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data());

 

             mask=Layer<Dtype>::binary_mask_adaptive[0]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);
             fft2(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());

             
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 

            if(frame_id==1)
            {
              count3=count3/sample_num;
            }

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample, num_per_channel1); 
        }
        else
        { 
             ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
             fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data(); 

             weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
             weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

             sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
             sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

             KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
             KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

             tmp_real1=Layer<Dtype>::first_layer_tmp_real1[blob_id]->mutable_gpu_data();
             tmp_imag1=Layer<Dtype>::first_layer_tmp_imag1[blob_id]->mutable_gpu_data();
  
             hf_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
             hf_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

             laplace_real=Layer<Dtype>::laplace_real[blob_id]->mutable_gpu_data();
             laplace_imag=Layer<Dtype>::laplace_imag[blob_id]->mutable_gpu_data();

             col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); num_per_channel1=row_num*(col_num/2+1);
             num_per_channel2=row_num*col_num; 
          
             count1=this->blobs_[blob_id]->channels()*row_num*(col_num/2+1);
             count2=this->blobs_[blob_id]->channels()*row_num*col_num;  
             count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
             number_per_sample=this->blobs_[blob_id]->channels()*(col_num/2+1)*row_num;
            ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
             ifft2(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
             scale_factor=col_num*row_num; 
             scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor); 

             set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_real);
             set_zeros<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, laplace_imag);

             mask=Layer<Dtype>::binary_mask[1]->mutable_gpu_data();
             add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);
             fft2(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
              
             fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),
             Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data());

                mask=Layer<Dtype>::binary_mask_adaptive[1]->mutable_gpu_data();
                add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);
                fft2(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
              
               fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),
               Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());

               set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
               set_zeros<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 
              

            if(frame_id==1)
            {
              count3=count3/sample_num;
            }

            my_weight_sample_kernel<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag,hf_real, hf_imag, weighted_sample_real,weighted_sample_imag,number_per_sample,num_per_channel1); 
        }
    }
  
   Dtype* L_index=Layer<Dtype>::L_index[0]->mutable_gpu_data(); 
   set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_real[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_real[0]->count(),Layer<Dtype>::sh_real[0]->mutable_gpu_data());
   set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_imag[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_imag[0]->count(),Layer<Dtype>::sh_imag[0]->mutable_gpu_data());
   Dtype* sh_real=Layer<Dtype>::sh_real[0]->mutable_gpu_data();
   Dtype* sh_imag=Layer<Dtype>::sh_imag[0]->mutable_gpu_data();
    
   for(int blob_id=0;blob_id<feature_num;blob_id++)
   {
       if(blob_id!=2)
       {
           caffe_gpu_add(Layer<Dtype>::sh_real[0]->count(),sh_real,Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(),sh_real);
           caffe_gpu_add(Layer<Dtype>::sh_imag[0]->count(),sh_imag,Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data(),sh_imag); 
       }
       else
       {
        int count7=Layer<Dtype>::first_layer_weighted_sample_real[0]->count();
           num_per_channel1=Layer<Dtype>::first_layer_hf_real[0]->height()*(Layer<Dtype>::first_layer_hf_real[0]->width());
           num_per_channel2=Layer<Dtype>::first_layer_hf_real[2]->height()*(Layer<Dtype>::first_layer_hf_real[2]->width());
           add_different_layers<<<CAFFE_GET_BLOCKS(count7), CAFFE_CUDA_NUM_THREADS>>>(count7,num_per_channel1, num_per_channel2, L_index, Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data(), sh_real, sh_imag);
       }

   }
   
Dtype* L_index1=Layer<Dtype>::L_index1[0]->mutable_gpu_data();
  for(int blob_id=0;blob_id<feature_num;blob_id++)
  {
     if(blob_id!=2)
     {
      count1=this->blobs_[blob_id]->channels()*Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      number_per_sample=num_per_channel1*this->blobs_[blob_id]->channels();
      KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data(); KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();
      sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
      sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();

      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_real);
      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_imag);

     weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, sh_real, sh_imag, KK_real,KK_imag,
                                                                                              sample_weight, number_per_sample,num_per_channel1, sample_num); 
     }
    else
    {
      count1=this->blobs_[blob_id]->channels()*Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      count2=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      num_per_channel2=Layer<Dtype>::first_layer_hf_real[0]->height()*Layer<Dtype>::first_layer_hf_real[0]->width();  
      number_per_sample=num_per_channel1*this->blobs_[blob_id]->channels();

      weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
      weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();


      KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data(); KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();

      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_real);
      set_zeros<<<CAFFE_GET_BLOCKS(Layer<Dtype>::KK_real[blob_id]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::KK_real[blob_id]->count(),KK_imag);

      sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
      sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data();
      crop_sample<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,num_per_channel1,num_per_channel2, L_index1, sh_real, sh_imag, Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(), Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data());  
     weight_sample_kernel_second<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, KK_real,KK_imag,sample_weight, number_per_sample,num_per_channel1, sample_num); 
    }
  }
Dtype lambda1=0.15; Dtype lambda2=1; Dtype lambda3=0.0;

for(int blob_id=0;blob_id<feature_num;blob_id++)
{
 if(blob_id!=2)
    {
      count1=Layer<Dtype>::KK_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();  
       fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data();
        row_num=Layer<Dtype>::KK_real[blob_id]->height(); col_num=row_num; 
        num_per_channel2=row_num*col_num;
        count2=num_per_channel2*this->blobs_[blob_id]->channels();
      ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor);
         mask=Layer<Dtype>::binary_mask_adaptive[0]->mutable_gpu_data();
         add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in2, this->d_in_tmp2);
        fft2(this->forward_plan[blob_id],this->d_in_tmp2,this->d_freq2);
         fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq2,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data()); 
       
        
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),lambda1*10,lambda2,this->blobs_[blob_id]->mutable_gpu_data()); 
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data()+count1,Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(),lambda1*10,lambda2,this->blobs_[blob_id]->mutable_gpu_data()+count1);
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data(),Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)0.01,this->blobs_[blob_id]->mutable_gpu_data()); 
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data()+count1,Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)0.01,this->blobs_[blob_id]->mutable_gpu_data()+count1);   
          
    }
    else
    {
       count1=Layer<Dtype>::KK_real[blob_id]->count();
      num_per_channel1=Layer<Dtype>::first_layer_hf_real[blob_id]->height()*Layer<Dtype>::first_layer_hf_real[blob_id]->width();
      ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();  
       fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data();
        row_num=Layer<Dtype>::KK_real[blob_id]->height(); col_num=row_num; 
        num_per_channel2=row_num*col_num;
        count2=num_per_channel2*this->blobs_[blob_id]->channels();
      ifftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
         ifft2(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
         scale_factor=col_num*row_num; 
         scale_out_real<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor);
         mask=Layer<Dtype>::binary_mask_adaptive[1]->mutable_gpu_data();
         add_mask<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,mask, this->d_in3, this->d_in_tmp3);
        fft2(this->forward_plan[blob_id],this->d_in_tmp3,this->d_freq3);
         fftshift<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,num_per_channel1,fftshift_mask,this->d_freq3,Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data());  
        
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data(),10*lambda1,lambda2,this->blobs_[blob_id]->mutable_gpu_data()); 
        caffe_gpu_add1(count1,this->blobs_[blob_id]->mutable_gpu_data()+count1,Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data(),10*lambda1,lambda2,this->blobs_[blob_id]->mutable_gpu_data()+count1);
        caffe_gpu_add2(count1,this->blobs_[blob_id]->mutable_gpu_data(),Layer<Dtype>::hf_tmp_real[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)0.3,this->blobs_[blob_id]->mutable_gpu_data(),frame_id); 
        caffe_gpu_add2(count1,this->blobs_[blob_id]->mutable_gpu_data()+count1,Layer<Dtype>::hf_tmp_imag[blob_id]->mutable_gpu_data(),(Dtype)1.0,(Dtype)0.3,this->blobs_[blob_id]->mutable_gpu_data()+count1,frame_id);    
        
    }

}



Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5) 
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4); 
 cudaFree(this->d_in_tmp1); cudaFree(this->d_in_tmp2); cudaFree(this->d_in_tmp3); cudaFree(this->d_in_tmp4); 
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);
  cudaFree(this->d_in_total1); cudaFree(this->d_in_total2);
   cudaFree(this->d_freq_total1); cudaFree(this->d_freq_total2);
   cudaFree(this->d_in_sub_total1); cudaFree(this->d_in_sub_total2);
   cudaFree(this->d_freq_sub_total1); cudaFree(this->d_freq_sub_total2);

cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]);
 cufftDestroy(this->forward_plan_total[0]); cufftDestroy(this->forward_plan_total[1]);
 cufftDestroy(this->forward_plan_sub_total[0]); cufftDestroy(this->forward_plan_sub_total[1]);  

cufftDestroy(this->inverse_plan[0]); cufftDestroy(this->inverse_plan[1]); cufftDestroy(this->inverse_plan[2]); cufftDestroy(this->inverse_plan[3]);
 cufftDestroy(this->inverse_plan_total[0]); cufftDestroy(this->inverse_plan_total[1]); 


   if(feature_num==5)
    { printf("the memory is released\n");
      cudaFree(this->d_in5);
      cudaFree(this->d_in_tmp5);  
      cudaFree(this->d_freq5); 
      cufftDestroy(this->forward_plan[4]);
      cufftDestroy(this->inverse_plan[4]);  
    }


}
   

}



template <typename Dtype>
void WtfLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {



}

INSTANTIATE_LAYER_GPU_FUNCS(WtfLayer);

}  // namespace caffe
