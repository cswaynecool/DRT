#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GMMLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  num_mixtures_ = this->layer_param_.gmm_param().num_mixtures();
  base_ = static_cast<Dtype>(this->layer_param_.gmm_param().base());
}


template <typename Dtype>
void GMMLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // Check mixture parameter dimension
  // mixture weights
  CHECK_EQ(num_mixtures_, bottom[0]->height()) << "Number of mixture weights should be equal to that of mixtures";
  // mu
  CHECK_EQ(2 * num_mixtures_, bottom[1]->height()) << "Number of means should be 2 times of that of mixtures";
  // sigma
  CHECK_EQ(2 * num_mixtures_, bottom[2]->height()) << "Number of variance should be 2 times of that of mixtures";
  // p
  CHECK_EQ(num_mixtures_, bottom[3]->height()) << "Number of correlations should be equal to that of mixtures";
  // label
  CHECK_EQ(2, bottom[4]->height()) << "Wrong label dimensions";
  num_ = bottom[0]->num();
  time_step_ = bottom[0]->channels();
  CHECK_EQ(num_, bottom[1]->num());
  CHECK_EQ(num_, bottom[2]->num());
  CHECK_EQ(num_, bottom[3]->num());
  CHECK_EQ(num_, bottom[4]->num());
  CHECK_EQ(time_step_, bottom[1]->channels());
  CHECK_EQ(time_step_, bottom[2]->channels());
  CHECK_EQ(time_step_, bottom[3]->channels());
  CHECK_EQ(time_step_, bottom[4]->channels());
  
  // reshape mixture blob N_ and overall probability P_
  N_.ReshapeLike(*bottom[0]);
  P_.Reshape(num_, time_step_, 1, 1);
  sum_ones_.ReshapeLike(P_);
  caffe_set(sum_ones_.count(), Dtype(1),
      sum_ones_.mutable_cpu_data());
}


template <typename Dtype>
void GMMLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(WARNING) << this->type() << "Layer does not implement Forward_cpu or Backward_cpu";
}

template <typename Dtype>
void GMMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(WARNING) << this->type() << "Layer does not implement Forward_cpu or Backward_cpu";
}

#ifdef CPU_ONLY
STUB_GPU(GMMLossLayer);
#endif

INSTANTIATE_CLASS(GMMLossLayer);
REGISTER_LAYER_CLASS(GMMLoss);

}  // namespace caffe
