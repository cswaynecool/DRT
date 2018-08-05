#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

void fft2_fourth(cufftHandle forward_plan, float* d_in, float2* d_freq)
{
    cufftExecR2C(forward_plan, d_in, d_freq);
    
}

void ifft2_fourth(cufftHandle inverse_plan, float2* d_freq, float* d_out)
{

     cufftExecC2R(inverse_plan, d_freq, d_out);
    
}

template <typename Dtype>
__global__ void ifftshift_fourth(const int n, int num_per_channel, Dtype* L_mask, Dtype* input_real, Dtype* input_imag, float2* output, int num_per_channel1) {
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
__global__ void fftshift_fourth(const int n, int num_per_channel1, Dtype* L_mask, float2* input, Dtype* output_real, Dtype* output_imag) {
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

__global__ void scale_out_real_fourth(const int n, float* input, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
  input[index]=input[index]/scale_factor;
  }
}

template <typename Dtype>   
__global__ void add_mask_fourth(const int n, int num_per_channel, Dtype* mask, float* input, float * output, int blob_id,int frame_id) {
  CUDA_KERNEL_LOOP(index, n) {
   int channel_index=index/num_per_channel;
   int current_index=index%num_per_channel;
  if(blob_id!=2&&frame_id>20)
 {
   output[index]=input[index]*mask[current_index]*2;
 }
else
      {
        output[index]=input[index]*mask[current_index]; 
      }



  }
}


template <typename Dtype>
void WtffourthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
   Dtype* data=Layer<Dtype>::feature_num[0]->mutable_cpu_data();
   int feature_num=data[0];
   Dtype* ifftshift_mask; Dtype* fftshift_mask;

   Dtype* frame_id_cpu=Layer<Dtype>::frame[0]->mutable_cpu_data();
   int frame_id=frame_id_cpu[0];

    for(int blob_id=0;blob_id<feature_num; blob_id++)
    {
        if(blob_id!=2)
        {
          ifftshift_mask=Layer<Dtype>::ifftshift_mask[0]->mutable_gpu_data();
          fftshift_mask=Layer<Dtype>::fftshift_mask[0]->mutable_gpu_data();  
          int count1=Layer<Dtype>::matlab_hf_real[blob_id]->count();
          Dtype* hf1_real=Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data();
          Dtype* hf1_imag=Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data();
          int num_per_channel1=Layer<Dtype>::matlab_hf_real[blob_id]->height()*Layer<Dtype>::matlab_hf_real[blob_id]->width();
          ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, hf1_real , hf1_imag, this->d_freq2, num_per_channel1);
          ifft2_fourth(this->inverse_plan[blob_id],this->d_freq2,this->d_in2);
          Dtype* binary_mask_adaptive=Layer<Dtype>::binary_mask_adaptive[0]->mutable_gpu_data();
          int num_per_channel2=Layer<Dtype>::matlab_hf_real[blob_id]->height()*Layer<Dtype>::matlab_hf_real[blob_id]->height();
          Dtype scale_factor=num_per_channel2;
          int count2=num_per_channel2*Layer<Dtype>::matlab_hf_real[blob_id]->channels();
          scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in2,scale_factor); 
          
           add_mask_fourth<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,binary_mask_adaptive, this->d_in2, this->d_in2,blob_id,frame_id);

           fft2_fourth(this->forward_plan[blob_id],this->d_in2,this->d_freq2);
            
           fftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, fftshift_mask, this->d_freq2, this->blobs_[blob_id]->mutable_gpu_data(), this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->offset(1)); 

        }
        else
        {
            
          ifftshift_mask=Layer<Dtype>::ifftshift_mask[1]->mutable_gpu_data();
          fftshift_mask=Layer<Dtype>::fftshift_mask[1]->mutable_gpu_data();  
          int count1=Layer<Dtype>::matlab_hf_real[blob_id]->count();
          Dtype* hf1_real=Layer<Dtype>::matlab_hf_real[blob_id]->mutable_gpu_data();
          Dtype* hf1_imag=Layer<Dtype>::matlab_hf_imag[blob_id]->mutable_gpu_data();
          int num_per_channel1=Layer<Dtype>::matlab_hf_real[blob_id]->height()*Layer<Dtype>::matlab_hf_real[blob_id]->width();
          ifftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, ifftshift_mask, hf1_real , hf1_imag, this->d_freq3, num_per_channel1);
          ifft2_fourth(this->inverse_plan[blob_id],this->d_freq3,this->d_in3);
          Dtype* binary_mask_adaptive=Layer<Dtype>::binary_mask_adaptive[1]->mutable_gpu_data();
          int num_per_channel2=Layer<Dtype>::matlab_hf_real[blob_id]->height()*Layer<Dtype>::matlab_hf_real[blob_id]->height();
          Dtype scale_factor=num_per_channel2;
          int count2=num_per_channel2*Layer<Dtype>::matlab_hf_real[blob_id]->channels();
          scale_out_real_fourth<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2,this->d_in3,scale_factor); 
          
           add_mask_fourth<<<CAFFE_GET_BLOCKS(count2), CAFFE_CUDA_NUM_THREADS>>>(count2, num_per_channel2,binary_mask_adaptive, this->d_in3, this->d_in3,blob_id,frame_id);
           fft2_fourth(this->forward_plan[blob_id],this->d_in3,this->d_freq3);
            
           fftshift_fourth<<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, num_per_channel1, fftshift_mask, this->d_freq3, this->blobs_[blob_id]->mutable_gpu_data(), this->blobs_[blob_id]->mutable_gpu_data()+this->blobs_[blob_id]->offset(1)); 



        }
    }

 Dtype* clear_memory_cpu=Layer<Dtype>::clear_memory[0]->mutable_cpu_data();
if(clear_memory_cpu[0]>0.5)
{
  cudaFree(this->d_in1); cudaFree(this->d_in2); cudaFree(this->d_in3); cudaFree(this->d_in4);  
  cudaFree(this->d_freq1); cudaFree(this->d_freq2); cudaFree(this->d_freq3); cudaFree(this->d_freq4);

cufftDestroy(this->forward_plan[0]); cufftDestroy(this->forward_plan[1]); cufftDestroy(this->forward_plan[2]); cufftDestroy(this->forward_plan[3]);

cufftDestroy(this->inverse_plan[0]); cufftDestroy(this->inverse_plan[1]); cufftDestroy(this->inverse_plan[2]); cufftDestroy(this->inverse_plan[3]);

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
void WtffourthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "start of convolutionlayer backward_gpu";
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin."
 const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
 
const unsigned int* mask = this->mask_.gpu_data();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* top_diff_mutable = top[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    // Mask
  //  if (this->has_mask_ && this->phase_ == TRAIN) {
  //    const unsigned int* mask = this->mask_.gpu_data();
  //    for (int n = 0; n < this->num_; ++n) {
  //  this->backward_gpu_mask(top_diff_mutable + top[i]->offset(n), mask);
  //    }
  //  }
   

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
	  if (this->kstride_h_ == 1) {
	    this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	  } else {
	    this->fcn_weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	    //LOG(INFO) << "fcn_weight_gpu_gemm";
	  }
        }
     this->backward_gpu_mask(weight_diff, mask); 

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtffourthLayer);

}  // namespace caffe
