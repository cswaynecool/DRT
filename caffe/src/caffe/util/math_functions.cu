#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <typename Dtype>
__global__ void add_kernel1(const int n, const Dtype* a,
    const Dtype* b, const Dtype lambda1, const Dtype lambda2, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index]*lambda1 + b[index]*lambda2;
  }
}

template <typename Dtype>
__global__ void add_kernel2(const int n, const Dtype* a,
    const Dtype* b, const Dtype lambda1, const Dtype lambda2, Dtype* y, int frame_id) {
  CUDA_KERNEL_LOOP(index, n) {
    if (frame_id<20)
      {
        y[index] = a[index]*lambda1 + b[index]*0.01;
      }
      else
      {
        y[index] = a[index]*lambda1 + b[index]*lambda2;

      }
    }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add1<float>(const int N, const float* a, const float* b, const float lambda1, const float lambda2,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel1<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b,lambda1,lambda2, y);
}

template <>
void caffe_gpu_add1<double>(const int N, const double* a, const double* b, const double lambda1, const double lambda2,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel1<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b,lambda1,lambda2, y);
}

template <>
void caffe_gpu_add2<float>(const int N, const float* a, const float* b, const float lambda1, const float lambda2,
    float* y, int frame_id) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel2<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b,lambda1,lambda2, y, frame_id);
}

template <>
void caffe_gpu_add2<double>(const int N, const double* a, const double* b, const double lambda1, const double lambda2,
    double* y, int frame_id) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel2<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b,lambda1,lambda2, y, frame_id);
}

template <typename Dtype>
__global__ void weight_sample_kernel(const int n, Dtype* sample_real, Dtype* sample_imag,
    Dtype* weight_real, Dtype* weight_imag, Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  CUDA_KERNEL_LOOP(index, n) {

   
   // int index1=index%number_per_sample;
   // int sample_index=index/number_per_sample;
   // int position_index=index1%number_per_channel+sample_index*number_per_channel;
   //  weighted_sample_real[index]=weight_real[index1]*sample_real[index]-weight_imag[index1]*sample_imag[index];
   //  weighted_sample_imag[index]=weight_real[index1]*sample_imag[index]+weight_imag[index1]*sample_real[index]; 
   
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

template <>
void caffe_gpu_weight_sample<float>(const int N, float* sample_real, float* sample_imag, float* weight_real, float* weight_imag,
    float* weighted_sample_real,float* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        weight_sample_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, sample_real, sample_imag, weight_real, weight_imag, weighted_sample_real, weighted_sample_imag, number_per_sample, number_per_channel);
}

template <>
void caffe_gpu_weight_sample<double>(const int N, double* sample_real, double* sample_imag, double* weight_real, double* weight_imag,
    double* weighted_sample_real, double* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        weight_sample_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, sample_real, sample_imag, weight_real, weight_imag, weighted_sample_real, weighted_sample_imag, number_per_sample, number_per_channel);
}


template <typename Dtype>
__global__ void sample_sum_kernel(const int n, Dtype* sample_real, Dtype* sample_imag,
  Dtype* weighted_sample_real,Dtype* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  CUDA_KERNEL_LOOP(index, n) {
 
    int channel_num=number_per_sample/number_per_channel;
    int sample_index=index/number_per_channel;
    int position_index=index%number_per_channel;
     weighted_sample_real[index]=0;
     weighted_sample_imag[index]=0; 
    for(int i=0;i<channel_num;i++)
    {
     int hf_base_position=position_index+i*number_per_channel;
     weighted_sample_real[index]=weighted_sample_real[index]+ sample_real[hf_base_position+number_per_sample*sample_index];
    weighted_sample_imag[index]= weighted_sample_imag[index]+sample_imag[hf_base_position+number_per_sample*sample_index];
    }
  


  }
}

template <>
void caffe_gpu_sample_sum<float>(const int N, float* sample_real, float* sample_imag,
    float* weighted_sample_real,float* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        sample_sum_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, number_per_sample, number_per_channel);
}

template <>
void caffe_gpu_sample_sum<double>(const int N, double* sample_real, double* sample_imag,
    double* weighted_sample_real, double* weighted_sample_imag, int number_per_sample,int number_per_channel) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        sample_sum_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, sample_real, sample_imag, weighted_sample_real, weighted_sample_imag, number_per_sample, number_per_channel);
}


template <typename Dtype>
__global__ void compute_diff_kernel(const int n,int number_per_sample1, int number_per_sample2,Dtype* L_mask, Dtype* real1, Dtype* real2, Dtype* real3, Dtype* real4, Dtype* real5,Dtype* imag1,Dtype* imag2, Dtype* imag3,Dtype* imag4,Dtype* imag5,Dtype* y_real,Dtype* y_imag,Dtype* y_diff_real, Dtype* y_diff_imag) {
  CUDA_KERNEL_LOOP(index, n) {
 //我们首先判断当前的index是第几个样本的
    int sample_index1=index/number_per_sample1;
    int index1=index%number_per_sample1;
    int index2=number_per_sample2*sample_index1+L_mask[index1]-1;
   if(L_mask[index1]==0)
      {
        y_diff_real[index]=real1[index]+real2[index]+real4[index]+real5[index]-y_real[index1];
        y_diff_imag[index]=imag1[index]+imag2[index]+imag4[index]+imag5[index]-y_imag[index1]; 
      }
   else
      { 
        y_diff_real[index]=real1[index]+real2[index]+real3[index2]+real4[index]+real5[index]-y_real[index1];
        y_diff_imag[index]=imag1[index]+imag2[index]+imag3[index2]+imag4[index]+imag5[index]-y_imag[index1]; 

      }
  }
}

template <>
void caffe_gpu_compute_diff<float>(const int N,int number_per_sample1, int number_per_sample2 ,float* L_mask, float* real1, float* real2, float* real3, float* real4, float* real5, float* imag1, float* imag2, float* imag3, float* imag4, float* imag5, float* y_real, float* y_imag,float* y_diff_real, float* y_diff_imag) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    compute_diff_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N,number_per_sample1,number_per_sample2, L_mask, real1, real2, real3, real4, real5, imag1, imag2,imag3, imag4, imag5, y_real, y_imag,y_diff_real,y_diff_imag);
}

template <>
void caffe_gpu_compute_diff<double>(const int N, int number_per_sample1, int number_per_sample2, double* L_mask, double* real1, double* real2, double* real3, double* real4, double* real5, double* imag1, double* imag2, double* imag3, double* imag4, double* imag5, double* y_real, double* y_imag,double* y_diff_real, double* y_diff_imag) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    compute_diff_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N,number_per_sample1,number_per_sample2,L_mask,real1, real2, real3, real4, real5, imag1, imag2,imag3, imag4, imag5, y_real, y_imag,y_diff_real,y_diff_imag);
}

template <typename Dtype>
__global__ void compute_gradient_kernel(const int n, Dtype* real, Dtype* imag,Dtype* yf_diff_real,Dtype* yf_diff_imag, Dtype* weight_diff, int num_per_sample,int num_per_channel,Dtype* sample_weight,int sample_num) {
  CUDA_KERNEL_LOOP(index, n) {
int index1=index%num_per_channel;
weight_diff[index]=0;
    for(int sample_index=0; sample_index<sample_num;sample_index++)
      { int index2=index1+sample_index*num_per_channel;
        weight_diff[index]=weight_diff[index]+sample_weight[sample_index]*(yf_diff_real[index2]*real[index+num_per_sample*sample_index]-yf_diff_imag[index2]*imag[index+num_per_sample*sample_index]);
        weight_diff[index+n]=weight_diff[index+n]+sample_weight[sample_index]*(yf_diff_real[index2]*imag[index+num_per_sample*sample_index]+yf_diff_imag[index2]*real[index+num_per_sample*sample_index]); 
      }
  
  
  }
}


template <>
void caffe_gpu_compute_gradient<float>(const int N, float* real, float* imag, float* yf_diff_real, float* yf_diff_imag, float* weight_diff, int num_per_sample,int num_per_channel,float* sample_weight,int sample_num) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    compute_gradient_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, real, imag,yf_diff_real,yf_diff_imag, weight_diff, num_per_sample, num_per_channel,sample_weight,sample_num);
}

template <>
void caffe_gpu_compute_gradient<double>(const int N, double* real, double* imag, double* yf_diff_real, double* yf_diff_imag, double* weight_diff, int num_per_sample,int num_per_channel, double* sample_weight,int sample_num) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    compute_gradient_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, real, imag,yf_diff_real,yf_diff_imag, weight_diff, num_per_sample, num_per_channel,sample_weight,sample_num);
}

// crop diff
template <typename Dtype>
__global__ void crop_diff_kernel(const int n,int number_per_channel1, int number_per_channel2,Dtype* L_mask,Dtype* y_diff_real, Dtype* y_diff_imag,Dtype* y_diff_cropped_real, Dtype* y_diff_cropped_imag) {
  CUDA_KERNEL_LOOP(index, n) {
    
    int sample_index1=index/number_per_channel1;
    int index1=index%number_per_channel1;
    if (L_mask[index1]!=0)
    {
      int index3=L_mask[index1]-1+sample_index1*number_per_channel2;
    y_diff_cropped_real[index3]=y_diff_real[index];
    y_diff_cropped_imag[index3]=y_diff_imag[index];
    } 
  
  }
}

template <>
void caffe_gpu_crop_diff<float>(const int N,int number_per_channel1, int number_per_channel2, float* L_mask,float* y_diff_real, float* y_diff_imag,float* y_diff_cropped_real, float* y_diff_cropped_imag) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    crop_diff_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, number_per_channel1,number_per_channel2,L_mask,y_diff_real,y_diff_imag,y_diff_cropped_real, y_diff_cropped_imag);
}

template <>
void caffe_gpu_crop_diff<double>(const int N,int number_per_channel1, int number_per_channel2, double* L_mask,double* y_diff_real, double* y_diff_imag, double* y_diff_cropped_real, double* y_diff_cropped_imag) {
  // NOLINT_NEXT_LINE(whitespace/operators)
    crop_diff_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, number_per_channel1, number_per_channel2,L_mask,y_diff_real,y_diff_imag,y_diff_cropped_real, y_diff_cropped_imag);
}

template <typename Dtype>
__global__ void set_zeros_kernel(const int n, Dtype* sample) {
  CUDA_KERNEL_LOOP(index, n) {
    sample[index]=0;
  
  }
}

template <>
void caffe_gpu_set_zeros<float>(const int N, float* sample) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        set_zeros_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            N, sample);
}

template <>
void caffe_gpu_set_zeros<double>(const int N, double* sample) {
  // NOLINT_NEXT_LINE(whitespace/operators)
        set_zeros_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
            N, sample);
}


template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <typename Dtype>
__global__ void my_mul_kernel(const int n, const Dtype* a,
    const unsigned int* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <typename Dtype>
__global__ void dtpooling_kernel(const int n, const Dtype* col_buff,
    Dtype* weight, Dtype* x, Dtype* x1, Dtype* y, Dtype* y1 , int conv_in_height_, int conv_in_width_, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    // 首先求得index的行及列坐标
    int width=conv_in_height_*conv_in_width_; 
   const int c = index/width;
   const int h = (index - c * width);//求得行index
   const int w = index % width; //求得列index
  output[index]=weight[0]*x[c]+weight[1]*x1[c]+weight[2]*y[c]+weight[3]*y1[c]+col_buff[index]; 
  //   output[index]=col_buff[index];
  }
}

template <typename Dtype>
__global__ void obtain_col_kernel(const int n, const Dtype* template1,
    int height, int width,int col_index ,Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    // 首先求得index的行及列坐标
   const int h = index/width;
   const int w = (index - h * width);//求得行index
      if(w==col_index)
      {
          output[h]=template1[index];
      }
  }
}


template <typename Dtype>
__global__ void refinenet_kernel(const int n, Dtype* col_buff,
    Dtype* weight, Dtype* x, Dtype* x1, Dtype* y, Dtype* y1 , int width, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    // 首先求得index的行及列坐标 
   const int c = index/width;
   const int h = (index - c * width);//求得行index
   const int w = index % width; //求得列index
  output[index]=weight[0]*x[c]+weight[1]*x1[c]+weight[2]*y[c]+weight[3]*y1[c]+col_buff[index]; 

  }
}

template <typename Dtype>
__global__ void place_col_kernel(const int n, Dtype* col_buff,
    int layer_index,int height, int width, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    // 当前index坐标为index列,layer_index行
    int current_index=index*width+layer_index;
    output[current_index]=col_buff[index];

  }
}

template <>
void caffe_gpu_placecol<float>(const int N, float* col_buff, int layer_index,int height, int width, float* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  place_col_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, layer_index, height, width, output);
}

template <>
void caffe_gpu_placecol<double>(const int N, double* col_buff, int layer_index,int height, int width, double* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  place_col_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, layer_index, height, width, output);
}


template <>
void caffe_gpu_refinenet<float>(const int N, float* col_buff, float* weight,float* x, float* x1, float* y, float* y1, int width, float* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  refinenet_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, weight, x, x1, y, y1, width, output);
}

template <>
void caffe_gpu_refinenet<double>(const int N, double* col_buff, double* weight,double* x, double* x1, double* y, double* y1, int width, double* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  refinenet_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, weight, x, x1, y, y1, width, output);
}

template <>
void caffe_gpu_obtain_col<float>(const int N, const float* template1, int height, int width, int col_index, float* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  obtain_col_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, template1, height, width, col_index, output);
}

template <>
void caffe_gpu_obtain_col<double>(const int N, const double* template1, int height, int width, int col_index, double* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  obtain_col_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, template1, height, width, col_index, output);
}


template <>
void caffe_gpu_dtpooling<float>(const int N, const float* col_buff, float* weight, float* x,
    float* x1, float* y, float* y1,int conv_in_height_, int conv_in_width_, float* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  dtpooling_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, weight, x, x1, y, y1,conv_in_height_,conv_in_width_ ,output);
}

template <>
void caffe_gpu_dtpooling<double>(const int N, const double* col_buff, double* weight, double* x,
    double* x1, double* y, double* y1,int conv_in_height_, int conv_in_width_, double* output) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  dtpooling_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, col_buff, weight, x, x1, y, y1,conv_in_height_,conv_in_width_,output);
}



template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <>
void caffe_gpu_mul1<float>(const int N, const float* a,
    const unsigned int* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  my_mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul1<double>(const int N, const double* a,
    const unsigned int* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  my_mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

__global__ void popc_kernel(const int n, const float* a,
    const float* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(a[index]) ^
                      static_cast<uint32_t>(b[index]));
  }
}

__global__ void popcll_kernel(const int n, const double* a,
    const double* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
                      static_cast<uint64_t>(b[index]));
  }
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        /* NOLINT_NEXT_LINE(build/include_what_you_use) */
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
