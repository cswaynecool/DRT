#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

void fft2_third(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_third(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

template <typename Dtype>
__global__ void ifftshift_third(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int row_num, int col_num,int num_per_channel1) {
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
__global__ void fftshift_third(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
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
      //output_real[index]=ori_index;
      //output_imag[index]=-ori_index; 
    }
  }
} 

__global__ void scale_out_real_third(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}

template <typename Dtype>
__global__ void set_zeros_third(const int n, Dtype* in_out) {
  CUDA_KERNEL_LOOP(index, n) {
  in_out[index]=0;
  }
}

template <typename Dtype>
void WtfthirdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
    Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
    int feature_num=data[0];
    set_zeros_third<<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS>>>(1, top[0]->mutable_gpu_data()); 
  
    Dtype scale_factor; 
  
   for(int blob_id=0; blob_id<feature_num;blob_id++)
    {
      if(blob_id!=2)
     {
      int col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); 
      int row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); int num_per_channel1=row_num*(col_num/2+1);
      int num_per_channel2=row_num*col_num;  int count1=Layer<Dtype>::first_layer_hf_real[blob_id]->count();
      int count3=col_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();
      Dtype* ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
      Dtype* fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data();   
      Dtype* xf_real=Layer<Dtype>::input_xff_real[blob_id]->mutable_gpu_data();
      Dtype* xf_imag=Layer<Dtype>::input_xff_imag[blob_id]->mutable_gpu_data();
      Dtype* yf_real=Layer<Dtype>::input_yff_real[blob_id]->mutable_gpu_data();
      Dtype* yf_imag=Layer<Dtype>::input_yff_imag[blob_id]->mutable_gpu_data();  
      ifftshift_third<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::input_xff_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::input_xff_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
           ifft2_third(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
        scale_factor=col_num*row_num; 
         scale_out_real_third<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in2,scale_factor);  
        ifftshift_third<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::input_yff_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::input_yff_imag[blob_id]->mutable_gpu_data(), this->d_freq2,row_num, col_num,num_per_channel1); 
           ifft2_third(this->inverse_plan[blob_id],this->d_freq2,this->d_in_tmp2);
        scale_out_real_third<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in_tmp2,scale_factor); 
         caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
         1,count3,
        (Dtype)row_num*col_num, (Dtype*)this->d_in2, (Dtype*)this->d_in_tmp2,
        (Dtype)1, top[0]->mutable_gpu_data());  
     }
      else
        {
            
      int col_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); 
      int row_num=Layer<Dtype>::first_layer_hf_real[blob_id]->height(); int num_per_channel1=row_num*(col_num/2+1);
      int num_per_channel2=row_num*col_num;  int count1=Layer<Dtype>::first_layer_hf_real[blob_id]->count();
      int count3=col_num*row_num*Layer<Dtype>::first_layer_hf_real[blob_id]->channels();
      Dtype* ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
      Dtype* fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data();   
      Dtype* xf_real=Layer<Dtype>::input_xff_real[blob_id]->mutable_gpu_data();
      Dtype* xf_imag=Layer<Dtype>::input_xff_imag[blob_id]->mutable_gpu_data();
      Dtype* yf_real=Layer<Dtype>::input_yff_real[blob_id]->mutable_gpu_data();
      Dtype* yf_imag=Layer<Dtype>::input_yff_imag[blob_id]->mutable_gpu_data();  
      ifftshift_third<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::input_xff_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::input_xff_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
           ifft2_third(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
         scale_factor=col_num*row_num; 
         scale_out_real_third<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in3,scale_factor);  
        ifftshift_third<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, Layer<Dtype>::input_yff_real[blob_id]->mutable_gpu_data() , Layer<Dtype>::input_yff_imag[blob_id]->mutable_gpu_data(), this->d_freq3,row_num, col_num,num_per_channel1); 
           ifft2_third(this->inverse_plan[blob_id],this->d_freq3,this->d_in_tmp3);
           scale_out_real_third<<<CAFFE_GET_BLOCKS(count3), CAFFE_CUDA_NUM_THREADS>>>(count3,this->d_in_tmp3,scale_factor);   
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
        1,count3,
        (Dtype)row_num*col_num, (Dtype*)this->d_in3, (Dtype*)this->d_in_tmp3,
        (Dtype)1, top[0]->mutable_gpu_data());  

        }



  } 

 Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5) 
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4); 
 cudaFree(this->d_in_tmp1); cudaFree(this->d_in_tmp2); cudaFree(this->d_in_tmp3); cudaFree(this->d_in_tmp4); 
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);
   cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]); 

    if(feature_num==5)
    {
      cudaFree(this->d_in5);
      cudaFree(this->d_in_tmp5);  
      cufftDestroy(this->forward_plan[4]);
      cufftDestroy(this->inverse_plan[4]);  
    }

} 


}
    
 
    
  

template <typename Dtype>
void WtfthirdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfthirdLayer);

}  // namespace caffe
