#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#define PI 3.1415926




namespace caffe { 

__device__ double squareD(double x) {
  return x * x;
}
__device__ float squareD(float x) {
  return x * x;
}

template <typename Dtype>
__global__ void ComputeN(const int nthreads, const Dtype* mean,
    const Dtype* variance, const Dtype* correlation, const Dtype* gt, 
    const unsigned int num_mixtures, Dtype* mixture_prob) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int sample_index = index / num_mixtures;
    const Dtype* mean_cur = mean + index * 2;
    const Dtype* variance_cur = variance + index * 2;
    const Dtype* correlation_cur = correlation + index;
    const Dtype* gt_cur = gt + sample_index * 2;
    Dtype* mixture_prob_cur = mixture_prob + index;
    
    Dtype normalize = Dtype(1) / (Dtype(2) * Dtype(PI) * variance_cur[0] * variance_cur[1] * sqrt(Dtype(1) - 
	  squareD(correlation_cur[0])));
    
    Dtype Z = squareD((gt_cur[0] - mean_cur[0]) / variance_cur[0]) +
      squareD((gt_cur[1] - mean_cur[1]) / variance_cur[1]) -
      Dtype(2) * correlation_cur[0] / variance_cur[0] / variance_cur[1] * 
      (gt_cur[0] - mean_cur[0]) * (gt_cur[1] - mean_cur[1]);

    mixture_prob_cur[0] = normalize * exp(-Z/Dtype(2)/(Dtype(1) - squareD(correlation_cur[0])));
  }
}

template <typename Dtype>
__global__ void ComputeP(const int nthreads, const Dtype* mixture_prob, 
    const Dtype* alpha, const unsigned int num_mixtures, const Dtype base, Dtype* prob) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype* mixture_prob_cur = mixture_prob + index * num_mixtures;
    const Dtype* alpha_cur = alpha + index * num_mixtures;
    Dtype* prob_cur = prob + index;
    *prob_cur = Dtype(base);

    for(int i = 0; i < num_mixtures; i++) {
      *prob_cur += alpha_cur[i] * mixture_prob_cur[i];
    }
  }
}

template <typename Dtype> 
__global__ void ComputeLogLike(const unsigned int nthreads, const Dtype* prob, Dtype* log_like) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype* prob_cur = prob + index;
    Dtype* log_like_cur = log_like + index;
    *log_like_cur = -log(*prob_cur);
  }
}

template <typename Dtype>
void GMMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype* alpha = bottom[0]->gpu_data();
  const Dtype* mean = bottom[1]->gpu_data();
  const Dtype* variance = bottom[2]->gpu_data();
  const Dtype* correlation = bottom[3]->gpu_data();
  const Dtype* label = bottom[4]->gpu_data();
  Dtype* mixture_prob = N_.mutable_gpu_data();
  Dtype* prob = P_.mutable_gpu_data();
  Dtype* log_like = P_.mutable_gpu_diff(); // store the log likelihood in P_.diff

  ComputeN<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, mean, variance, correlation, label, num_mixtures_, mixture_prob);
  count /= num_mixtures_; 
  ComputeP<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, mixture_prob, alpha, num_mixtures_, base_, prob);
  ComputeLogLike<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, prob, log_like);
  Dtype loss = Dtype(0);
 
  //caffe_gpu_asum(count, log_like, &loss);
  const Dtype* sum_ones = sum_ones_.gpu_data();
  caffe_gpu_dot(P_.count(), log_like, sum_ones, &loss);


  //top[0]->mutable_cpu_data()[0] = loss / Dtype(num_) / Dtype(time_step_);
  
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void GMMBackward(const unsigned int nthreads, const Dtype* alpha, const Dtype* mean, 
    const Dtype* variance, const Dtype* correlation, const Dtype* label, const Dtype* mixture_prob, 
    const Dtype* prob, const unsigned int num_mixtures, const Dtype loss_weight, 
    Dtype* alpha_diff, Dtype* mean_diff, Dtype* variance_diff, Dtype* correlation_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sample_index = index / num_mixtures;
    const Dtype* alpha_cur = alpha + index;
    const Dtype* mean_cur = mean + 2 * index;
    const Dtype* variance_cur = variance + 2 * index;
    const Dtype* correlation_cur = correlation + index;
    const Dtype* label_cur = label + 2 * sample_index;
    const Dtype* mixture_prob_cur = mixture_prob + index;
    const Dtype* prob_cur = prob + sample_index;
    Dtype* alpha_diff_cur = alpha_diff + index;
    Dtype* mean_diff_cur = mean_diff + 2 * index;
    Dtype* variance_diff_cur = variance_diff + 2 * index;
    Dtype* correlation_diff_cur = correlation_diff + index;
    // alpha_diff 
    alpha_diff_cur[0] = -loss_weight * mixture_prob_cur[0] / prob_cur[0];
    //alpha_diff_cur[0] = -2 * mixture_prob_cur[0] / prob_cur[0];
    Dtype t1, t2, t3;
    // mean1_diff
    t1 = alpha_diff_cur[0] * alpha_cur[0];
    t2 = Dtype(1) / variance_cur[0] / (Dtype(1) - squareD(correlation_cur[0]));
    t3 = (label_cur[0] - mean_cur[0]) / variance_cur[0] - 
      correlation_cur[0] * (label_cur[1] - mean_cur[1]) / variance_cur[1];
    mean_diff_cur[0] = t1 * t2 * t3;
    // mean2_diff
    t2 = Dtype(1) / variance_cur[1] / (Dtype(1) - squareD(correlation_cur[0]));
    t3 = (label_cur[1] - mean_cur[1]) / variance_cur[1] - 
      correlation_cur[0] * (label_cur[0] - mean_cur[0]) / variance_cur[0];
    mean_diff_cur[1] = t1 * t2 * t3;
    // variance1_diff
    t2 = Dtype(1) / variance_cur[0] / (Dtype(1) - squareD(correlation_cur[0])) * (label_cur[0] - mean_cur[0]);
    t3 = ((label_cur[0] - mean_cur[0]) / variance_cur[0] - 
	correlation_cur[0] * (label_cur[1] - mean_cur[1]) / variance_cur[1]);
    variance_diff_cur[0] = (t1 * (t2 * t3 - Dtype(1))) / variance_cur[0];
    // varaince2_dif
    t2 = Dtype(1) / variance_cur[1] / (Dtype(1) - squareD(correlation_cur[0])) * (label_cur[1] - mean_cur[1]);
    t3 = ((label_cur[1] - mean_cur[1]) / variance_cur[1] - 
	correlation_cur[0] * (label_cur[0] - mean_cur[0]) / variance_cur[0]);
    variance_diff_cur[1] = (t1 * (t2 * t3 - Dtype(1))) / variance_cur[1];
    // correlation_diff
    t2 = (label_cur[0] - mean_cur[0]) * (label_cur[1] - mean_cur[1]) / variance_cur[0] / variance_cur[1];
    
    Dtype Z = squareD((label_cur[0] - mean_cur[0]) / variance_cur[0]) +
      squareD((label_cur[1] - mean_cur[1]) / variance_cur[1]) -
      Dtype(2) * correlation_cur[0] / variance_cur[0] / variance_cur[1] * 
      (label_cur[0] - mean_cur[0]) * (label_cur[1] - mean_cur[1]);

    t3 = correlation_cur[0] * (Dtype(1) - Dtype(1) /(Dtype(1) - squareD(correlation_cur[0])) * Z);
    correlation_diff_cur[0] = t1 * (t2 + t3) / (Dtype(1) - squareD(correlation_cur[0]));
  }
}

template <typename Dtype>
void GMMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
  int count = bottom[0]->count();
  const Dtype* alpha = bottom[0]->gpu_data();
  const Dtype* mean = bottom[1]->gpu_data();
  const Dtype* variance = bottom[2]->gpu_data();
  const Dtype* correlation = bottom[3]->gpu_data();
  const Dtype* label = bottom[4]->gpu_data();
  const Dtype* mixture_prob = N_.gpu_data();
  const Dtype* prob = P_.gpu_data();
  const Dtype loss_weight = top[0]->cpu_diff()[0];

  Dtype* alpha_diff = bottom[0]->mutable_gpu_diff();
  Dtype* mean_diff = bottom[1]->mutable_gpu_diff();
  Dtype* variance_diff = bottom[2]->mutable_gpu_diff();
  Dtype* correlation_diff = bottom[3]->mutable_gpu_diff();
  GMMBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, alpha, mean, variance, correlation, label, mixture_prob, prob,
      num_mixtures_, loss_weight,
      alpha_diff, mean_diff, variance_diff, correlation_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(GMMLossLayer);

}  // namespace caffe
