// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out) {
  // f(x) = 0.5 * x^2    if |x| < 1
  //        |x| - 0.5    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = 0.5 * val * val;
    } else {
      out[index] = abs_val - 0.5;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();


  //printf ("SmoothL1LossLayer::Forward_gpu()\n");
  Blob<Dtype>* bbox_targets = bottom[1];
  Blob<Dtype>* bbox_loss_weights = bottom[2];
  Blob<Dtype>* labels = bottom[4];

  int DEBUG = 0;

  if (DEBUG == 1) {
	  int label_pos_no = 0;
	  int label_zero_no = 0;
	  int label_neg_no = 0;
	  int bbox_target_zero_no = 0;
	  int bbox_target_non_zero_no = 0;
	  int bbox_loss_weight_pos_no = 0;
	  int bbox_loss_weight_non_pos_no = 0;
  
	  for (int i = 0; i<2; i++)
		for (int j=0; j<36; j++)
		  for (int k=0; k<bbox_targets->height(); k++)
			  for (int l=0; l<bbox_targets->width(); l++) {
				Dtype bbox_target = bbox_targets->data_at(i, j, k, l);
				Dtype bbox_loss_weight = bbox_loss_weights->data_at(i, j, k, l);
			
				if (bbox_target != 0)
					bbox_target_non_zero_no++;
				else
					bbox_target_zero_no++;

				if (bbox_loss_weight > 0)
					bbox_loss_weight_pos_no++;
				else
					bbox_loss_weight_non_pos_no++;
			}

	  int value_match_no = 0;
  
	  for (int i = 0; i<2; i++)
	    for (int j=0; j<9; j++)
		  for (int k=0; k<bbox_targets->height(); k++)
			for (int l=0; l<bbox_targets->width(); l++) {
				Dtype label = labels->data_at(i, j, k, l);
			
				if (label == 1)
					label_pos_no++;
				else if (label == 0)
					label_zero_no++;
				else if (label < 0)
					label_neg_no++;

				Dtype bbox_target1 = bbox_targets->data_at(i, j*4, k, l);
				Dtype bbox_target2 = bbox_targets->data_at(i, j*4+1, k, l);
				Dtype bbox_target3 = bbox_targets->data_at(i, j*4+2, k, l);
				Dtype bbox_target4 = bbox_targets->data_at(i, j*4+3, k, l);
				Dtype bbox_loss_weight1 = bbox_loss_weights->data_at(i, j*4, k, l);
				Dtype bbox_loss_weight2 = bbox_loss_weights->data_at(i, j*4+1, k, l);
				Dtype bbox_loss_weight3 = bbox_loss_weights->data_at(i, j*4+2, k, l);
				Dtype bbox_loss_weight4 = bbox_loss_weights->data_at(i, j*4+3, k, l);

				if (label == 1){
					if (bbox_target1 != 0 && bbox_target2 != 0 && bbox_target3 != 0 && bbox_target4 != 0 && 
						bbox_loss_weight1 == 1 && bbox_loss_weight2 == 1 && bbox_loss_weight3 == 1 && bbox_loss_weight4 == 1)
						value_match_no++;
				}
			}

	  printf ("label_pos_no : %d\n", label_pos_no);
	  printf ("label_zero_no : %d\n", label_zero_no);
	  printf ("label_neg_no : %d\n", label_neg_no);
	  printf ("bbox_target_non_zero_no : %d\n", bbox_target_non_zero_no);
	  printf ("bbox_target_zero_no : %d\n", bbox_target_zero_no);
	  printf ("bbox_loss_weight_non_pos_no : %d\n", bbox_loss_weight_non_pos_no);
	  printf ("bbox_loss_weight_pos_no : %d\n", bbox_loss_weight_pos_no);
	  printf ("value_match_no : %d\n", value_match_no);
	  printf ("SmoothL1LossLayer count : %d\n", count);
	  printf ("batch_size_ : %d\n", batch_size_);
  }


  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w * (b0 - b1)
  }
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_asum(count, errors_.gpu_data(), &loss);
  
  if (batch_size_ > 0)
    top[0]->mutable_cpu_data()[0] = loss / batch_size_;
  else
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out) {
  // f'(x) = x         if |x| < 1
  //       = sign(x)   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1) {
      out[index] = val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      
	  // DJDJ
	  Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();

	  if (batch_size_ > 0)
		alpha = sign * top[0]->cpu_diff()[0] / batch_size_;

      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
