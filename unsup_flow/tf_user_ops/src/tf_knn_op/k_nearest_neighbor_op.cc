// Copyright 2022 MBition GmbH
// SPDX-License-Identifier: MIT

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "k_nearest_neighbor_op.h"

namespace tensorflow {

template <typename Device, typename Dtype>
class KNearestNeighborOp : public OpKernel {
 public:
  explicit KNearestNeighborOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_ref = context->input(0);
    const Tensor& tensor_query = context->input(1);
    auto input_query = tensor_query.tensor<Dtype, 2>();

    if (!context->status().ok()) {
      return;
    }

    const std::int32_t num_query_pts = input_query.dimension(0);

    Tensor* ot_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{{num_query_pts, k_}},
                                            &ot_indices));
    Tensor* ot_distances = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape{{num_query_pts, k_}},
                                            &ot_distances));

    ::tensorflow::functor::KNNFunctor<Device, Dtype> knn;
    knn(context, tensor_ref, tensor_query, ot_distances, ot_indices, k_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KNearestNeighborOp);

  std::int32_t k_;
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(KNearestNeighbor, CPU, float);

}  // namespace tensorflow
