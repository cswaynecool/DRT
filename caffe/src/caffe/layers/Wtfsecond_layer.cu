#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>   
__global__ void add_mask_H(const int n, Dtype* H_input, Dtype* patch_mask, Dtype* H_total, int num_per_channel, int num_per_sample) {
  CUDA_KERNEL_LOOP(index, n) {
  int current_index=index%num_per_channel;
  int current_index1=index%num_per_sample;
  H_total[index]=H_input[current_index1]*patch_mask[current_index];



  }
}

void fft2_second(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_second(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

 template <typename Dtype>
__global__ void copy_freq(const int n, Dtype* real, Dtype* imag, float2* output) {
  CUDA_KERNEL_LOOP(index, n) {
  output[index].x=real[index];
  output[index].y=imag[index];
  }
}

 template <typename Dtype>
__global__ void fftshift_second(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel1;
   int current_index=index%num_per_channel1;

   if(L_mask[current_index]>-0.5)
    {
      int ori_index=L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=input[ori_index].y;
     // output_real[index]=ori_index;
     // output_imag[index]=ori_index;     
    }
    else
    {
      int ori_index=-L_mask[current_index]+channel_index*num_per_channel1;
      output_real[index]=input[ori_index].x;
      output_imag[index]=-input[ori_index].y; 
    }
  }
}      

__global__ void scale_out_real_second(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}



template <typename Dtype>
__global__ void my_weight_sample_kernel1(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weight_real, Dtype* weight_imag, Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
 CUDA_KERNEL_LOOP(index, n) { 
    int channel_num=number_per_sample/number_per_channel;
    int sample_index=index/number_per_channel;
    int position_index=index%number_per_channel;
    for(int i=0;i<channel_num;i++)
    {int hf_base_position=position_index+i*number_per_channel;
     weighted_sample_real[index]= weighted_sample_real[index]+weight_real[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index]+weight_imag[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index];
    weighted_sample_imag[index]= weighted_sample_imag[index]+weight_real[hf_base_position]*sample_imag[hf_base_position+number_per_sample*sample_index]-weight_imag[hf_base_position]*sample_real[hf_base_position+number_per_sample*sample_index];
    } 

  }
}

template <typename Dtype>
__global__ void weight_sample_kernel_second1(const int n, Dtype* sample_real, Dtype* sample_imag,
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
__global__ void ifftshift_second(const int n, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
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
__global__ void get_freq_second(const int n, float2* freq, Dtype* top_data_real) {
  CUDA_KERNEL_LOOP(index, n) {
  top_data_real[index]=freq[index].x; 

  }
}

template <typename Dtype>
__global__ void set_zeros1(const int n, Dtype* in_out) {
  CUDA_KERNEL_LOOP(index, n) {
  in_out[index]=0;
  }
}

template <typename Dtype>
__global__ void add_different_layers1(const int n,int num_per_channel1, int num_per_channel2, Dtype* L_mask, Dtype* real,Dtype* imag, Dtype* sh_real, Dtype* sh_imag) {
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
__global__ void get_response(const int n, Dtype* reliability_response, Dtype* reliability_response1, Dtype* reliability_response2, Dtype* sample_weight,int sample_num,int num_per_channel3) {
CUDA_KERNEL_LOOP(index, n) {
  
    int sample_index=index/num_per_channel3;
    reliability_response1[index]=reliability_response[index]*sqrt(sample_weight[sample_index]);
    reliability_response2[index]=reliability_response[index]*sample_weight[sample_index];




  }
}

template <typename Dtype>
__global__ void test(const int n, Dtype* reliability_response2) {
CUDA_KERNEL_LOOP(index, n) {

    reliability_response2[index]=1;


  }
}

template <typename Dtype>
void WtfsecondLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];

    int frag_num=Layer<Dtype>::patch_mask[0]->channels();
    int count_real; int count_imag;
    int num_per_channel; int num_per_channel1; int num_per_channel2;
    int num_per_sample;  Dtype* ifftshift_mask; Dtype* fftshift_mask;  Dtype* sample_real; Dtype* sample_imag;
    Dtype* weight_real; Dtype* weight_imag; Dtype* weighted_sample_real; Dtype* weighted_sample_imag; int row_num; int col_num; float scale_factor;
    Dtype* patch_mask;
    Dtype* KK_real; Dtype* KK_imag;
    int number_per_sample;

    int num_per_channel3;  int num_per_sample1;


    Dtype* sample_weight=Layer<Dtype>::sample_weight[0]->mutable_gpu_data();

    int sample_num=Layer<Dtype>::sample_weight[0]->width();

    int count3;

    Dtype* L_index=Layer<Dtype>::L_index[0]->mutable_gpu_data(); 

  for(int frag_id=0;frag_id<frag_num;frag_id++)
  {        
    for(int blob_id=0;blob_id<feature_num; blob_id++)
      {  
        if(blob_id!=2)
        {
           sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
           sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data(); 

            weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
            weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

           weight_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
           weight_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

           KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
           KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();


           num_per_channel=Layer<Dtype>::filter_H[blob_id]->height()*Layer<Dtype>::filter_H[blob_id]->width();
           num_per_channel1=Layer<Dtype>::filter_H[blob_id]->height()*(Layer<Dtype>::filter_H[blob_id]->width()/2+1);

           number_per_sample=num_per_channel1*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();


           count_real=Layer<Dtype>::filter_H[blob_id]->count();
           count_imag=num_per_channel1*Layer<Dtype>::filter_H[blob_id]->channels();
           num_per_sample=num_per_channel*Layer<Dtype>::H_total[blob_id]->channels();
           num_per_sample1=num_per_channel1*Layer<Dtype>::H_total[blob_id]->channels();

           fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data(); 
           ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
           Dtype* patch_mask=Layer<Dtype>::patch_mask[0]->mutable_gpu_data()+num_per_channel*frag_id;
           Dtype* H_input=Layer<Dtype>::filter_H[blob_id]->mutable_gpu_data(); 
            
           add_mask_H<<<CAFFE_GET_BLOCKS(count_real), CAFFE_CUDA_NUM_THREADS>>>(count_real, H_input, patch_mask, (Dtype*)this->d_in2, num_per_channel, num_per_sample);
           fft2_second(this->forward_plan[blob_id],this->d_in2,this->d_freq2);
           fftshift_second<<<CAFFE_GET_BLOCKS(count_imag), CAFFE_CUDA_NUM_THREADS>>>(count_imag,num_per_channel1,fftshift_mask,this->d_freq2,weight_real, weight_imag);
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          set_zeros1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
          set_zeros1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 
   
          my_weight_sample_kernel1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag, weight_real, weight_imag, weighted_sample_real, weighted_sample_imag, num_per_sample1,num_per_channel1);  

        }
        else
        {
                  
           sample_real=Layer<Dtype>::first_layer_samplef_real[blob_id]->mutable_gpu_data();
           sample_imag=Layer<Dtype>::first_layer_samplef_imag[blob_id]->mutable_gpu_data(); 

            weighted_sample_real=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data();
            weighted_sample_imag=Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data();

           weight_real=Layer<Dtype>::first_layer_hf_real[blob_id]->mutable_gpu_data();
           weight_imag=Layer<Dtype>::first_layer_hf_imag[blob_id]->mutable_gpu_data();

           KK_real=Layer<Dtype>::KK_real[blob_id]->mutable_gpu_data();
           KK_imag=Layer<Dtype>::KK_imag[blob_id]->mutable_gpu_data();


           num_per_channel=Layer<Dtype>::filter_H[blob_id]->height()*Layer<Dtype>::filter_H[blob_id]->width();
           num_per_channel1=Layer<Dtype>::filter_H[blob_id]->height()*(Layer<Dtype>::filter_H[blob_id]->width()/2+1);

           number_per_sample=num_per_channel1*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();

           num_per_sample1=num_per_channel1*Layer<Dtype>::H_total[blob_id]->channels();

           count_real=Layer<Dtype>::filter_H[blob_id]->count();
           count_imag=num_per_channel1*Layer<Dtype>::filter_H[blob_id]->channels();
           num_per_sample=num_per_channel*Layer<Dtype>::H_total[blob_id]->channels();
           fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data(); 
           ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
           Dtype* patch_mask=Layer<Dtype>::patch_mask[1]->mutable_gpu_data()+num_per_channel*frag_id;
           Dtype* H_input=Layer<Dtype>::filter_H[blob_id]->mutable_gpu_data(); 
            
           add_mask_H<<<CAFFE_GET_BLOCKS(count_real), CAFFE_CUDA_NUM_THREADS>>>(count_real, H_input, patch_mask, (Dtype*)this->d_in3, num_per_channel, num_per_sample);
           fft2_second(this->forward_plan[blob_id],this->d_in3,this->d_freq3);
           fftshift_second<<<CAFFE_GET_BLOCKS(count_imag), CAFFE_CUDA_NUM_THREADS>>>(count_imag,num_per_channel1,fftshift_mask,this->d_freq3,weight_real, weight_imag);
          count3=Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->count();
          set_zeros1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_real);
          set_zeros1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,weighted_sample_imag); 
   
          my_weight_sample_kernel1<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3, sample_real, sample_imag, weight_real, weight_imag, weighted_sample_real, weighted_sample_imag, num_per_sample1,num_per_channel1);  

        }
 
    }

    
    set_zeros1<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_real[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_real[0]->count(),Layer<Dtype>::sh_real[0]->mutable_gpu_data());
    set_zeros1<<<CAFFE_GET_BLOCKS(Layer<Dtype>::sh_imag[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(Layer<Dtype>::sh_imag[0]->count(),Layer<Dtype>::sh_imag[0]->mutable_gpu_data());
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
           
       add_different_layers1<<<CAFFE_GET_BLOCKS(count7), CAFFE_CUDA_NUM_THREADS>>>(count7,num_per_channel1, num_per_channel2, L_index, Layer<Dtype>::first_layer_weighted_sample_real[blob_id]->mutable_gpu_data(),Layer<Dtype>::first_layer_weighted_sample_imag[blob_id]->mutable_gpu_data(), sh_real, sh_imag);
       }



   }

count_imag=Layer<Dtype>::sh_real[0]->count();
ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
num_per_channel1=Layer<Dtype>::filter_H[0]->height()*(Layer<Dtype>::filter_H[0]->width()/2+1);
ifftshift_second<<<CAFFE_GET_BLOCKS(count_imag), CAFFE_CUDA_NUM_THREADS>>>(count_imag, ifftshift_mask, Layer<Dtype>::sh_real[0]->mutable_gpu_data(), Layer<Dtype>::sh_imag[0]->mutable_gpu_data(),this->d_freq_response,Layer<Dtype>::sh_real[0]->height(), Layer<Dtype>::sh_real[0]->width(),num_per_channel1);
ifft2_second(this->inverse_plan_response[0], this->d_freq_response, (float*)Layer<Dtype>::reliability_response[0]->mutable_gpu_data());
scale_factor=Layer<Dtype>::sh_real[0]->height()*Layer<Dtype>::sh_real[0]->height(); 
count_real=Layer<Dtype>::sh_real[0]->height()*Layer<Dtype>::sh_real[0]->height()*Layer<Dtype>::sh_real[0]->num();
scale_out_real_second<<<CAFFE_GET_BLOCKS(count_real), CAFFE_CUDA_NUM_THREADS>>>(count_real,(float*)Layer<Dtype>::reliability_response[0]->mutable_gpu_data(), scale_factor);

Dtype* reliability_response=Layer<Dtype>::reliability_response[0]->mutable_gpu_data();
Dtype* reliability_response1=Layer<Dtype>::reliability_response1[0]->mutable_gpu_data();
Dtype* reliability_response2=Layer<Dtype>::reliability_response2[0]->mutable_gpu_data();

count_real=Layer<Dtype>::reliability_response[0]->count()/frag_num;
num_per_channel3=Layer<Dtype>::reliability_response[0]->height()*Layer<Dtype>::reliability_response[0]->width();

get_response<<<CAFFE_GET_BLOCKS(count_real), CAFFE_CUDA_NUM_THREADS>>>(count_real, reliability_response, reliability_response1+Layer<Dtype>::reliability_response1[0]->offset(frag_id), reliability_response2+Layer<Dtype>::reliability_response2[0]->offset(frag_id), sample_weight, sample_num,num_per_channel3);

   top[0]->Reshape(Layer<Dtype>::reliability_response[0]->num()*2,Layer<Dtype>::reliability_response[0]->channels(),Layer<Dtype>::reliability_response[0]->height(),Layer<Dtype>::reliability_response[0]->width());
   caffe_copy(top[0]->count()/2,reliability_response1,top[0]->mutable_gpu_data());
   caffe_copy(top[0]->count()/2,reliability_response2,top[0]->mutable_gpu_data()+top[0]->count()/2); 

  }

Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5) 
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4);  
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);

 cudaFree(this->d_in_response); cudaFree(this->d_freq_response); 

cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]);

cufftDestroy(this->inverse_plan[0]); cufftDestroy(this->inverse_plan[1]); cufftDestroy(this->inverse_plan[2]); cufftDestroy(this->inverse_plan[3]);

 cufftDestroy(this->forward_plan_response[0]);
 cufftDestroy(this->inverse_plan_response[0]);

   if(feature_num==5)
    { printf("the memory is released\n");
      cudaFree(this->d_in5);
      cudaFree(this->d_freq5); 
      cufftDestroy(this->forward_plan[4]);
      cufftDestroy(this->inverse_plan[4]);  
    }


}



}

template <typename Dtype>
void WtfsecondLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(WtfsecondLayer);

}  // namespace caffe
