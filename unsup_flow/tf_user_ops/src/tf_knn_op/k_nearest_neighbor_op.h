/*
 * Copyright 2022 MBition GmbH
 * SPDX-License-Identifier: MIT
 */

#ifndef K_NEAREST_NEIGHBOR_KERNELS_K_NEAREST_NEIGHBOR_OP_H_
#define K_NEAREST_NEIGHBOR_KERNELS_K_NEAREST_NEIGHBOR_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
    class OpKernelContext;
    class Tensor;
    using CPUDevice = Eigen::ThreadPoolDevice;
    using GPUDevice = Eigen::GpuDevice;

}
namespace tensorflow{
namespace functor {

template <typename Device, typename Dtype>
struct KNNFunctor {
    void operator()(::tensorflow::OpKernelContext* context,
                     const Tensor& tf_ref_pts, const Tensor& tf_query_pts,
                     Tensor* tf_distances, Tensor* indices, std::int32_t k);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // K_NEAREST_NEIGHBOR_KERNELS_MATRIX_ADD_OP_H_
