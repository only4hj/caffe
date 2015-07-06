#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

typedef ::testing::Types<FloatCPU, DoubleCPU> TestDtypesCPU;

template <typename TypeParam>
class SoftmaxLayer2Test : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayer2Test()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxLayer2Test() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayer2Test, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayer2Test, TestForward) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.mutable_softmax_param()->set_range(2);
	SoftmaxLayer2<Dtype> layer(layer_param);

	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	// Test sum
	for (int i = 0; i < this->blob_bottom_->num(); ++i) {
		for (int k = 0; k < this->blob_bottom_->height(); ++k) {
			for (int l = 0; l < this->blob_bottom_->width(); ++l) {
				for (int m = 0; m < 5; ++m) {
					Dtype sum = 0;
					for (int j = 0; j < 2; ++j) {
						sum += this->blob_top_->data_at(i, m * 2 + j, k, l);
					}
					EXPECT_GE(sum, 0.999);
					EXPECT_LE(sum, 1.001);
					// Test exact values
					Dtype scale = 0;
					for (int j = 0; j < 2; ++j) {
						scale += exp(this->blob_bottom_->data_at(i, m * 2 + j, k, l));
					}
					for (int j = 0; j < 2; ++j) {
						EXPECT_GE(this->blob_top_->data_at(i, m * 2 + j, k, l) + 1e-4,
							exp(this->blob_bottom_->data_at(i, m * 2 + j, k, l)) / scale)
							<< "debug: " << i << " " << m * 2 + j;
						EXPECT_LE(this->blob_top_->data_at(i, m * 2 + j, k, l) - 1e-4,
							exp(this->blob_bottom_->data_at(i, m * 2 + j, k, l)) / scale)
							<< "debug: " << i << " " << m * 2 + j;
					}
				}
			}
		}
	}
}

/*
TYPED_TEST(SoftmaxLayer2Test, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer2<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNSoftmaxLayer2Test : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNSoftmaxLayer2Test()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayer2Test() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayer2Test, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayer2Test, TestForwardCuDNN) {
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        TypeParam sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        TypeParam scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayer2Test, TestGradientCuDNN) {
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif
*/

}  // namespace caffe
