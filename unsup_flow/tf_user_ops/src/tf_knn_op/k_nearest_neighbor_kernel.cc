// Copyright 2022 MBition GmbH
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>

#include "k_nearest_neighbor_op.h"
#include "tensorflow/core/framework/op.h"

#include "nanoflann.hpp"

template <typename Derived> struct PointCloudAdaptor {
  const Derived &obj; //!< A const ref to the data set origin

  /// The constructor that sets the data set source
  PointCloudAdaptor(const Derived &obj_) : obj(obj_) {}

  /// CRTP helper method
  inline const Derived &derived() const { return obj; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const {
    return derived().dimensions()[0];
  }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    //// std::cout << "Acessing pt " << idx << " dim " << dim << std::endl;
    return derived()(idx, dim);
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to re do it again. Look at bb.size() to find
  //   out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }

}; // end of PointCloudAdaptor

namespace tensorflow {
namespace functor {

template <typename Dtype> struct KNNFunctor<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext *context,
                  const Tensor &tf_ref_pts, const Tensor &tf_query_pts,
                  Tensor *tf_distances, Tensor *tf_indices, std::int32_t k) {

    const auto ref_pts_tensor = tf_ref_pts.matrix<float>();

    const auto num_query_pts = tf_query_pts.dim_size(0);

    const auto query_pts_tensor = tf_query_pts.matrix<float>();

    auto eig_dists = tf_distances->tensor<float, 2>();
    auto eig_indices = tf_indices->tensor<int32_t, 2>();

    typedef PointCloudAdaptor<const Eigen::TensorMap<
        Eigen::Tensor<const float, 2, 1, Eigen::DenseIndex>, 16>>
        PC2KD;
    const PC2KD pc2kd(ref_pts_tensor); // The adaptor

    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PC2KD>, PC2KD, 3>
        my_kd_tree_t;

    my_kd_tree_t index(3 /*dim*/, pc2kd,
                       nanoflann::KDTreeSingleIndexAdaptorParams(20));

    index.buildIndex();
    std::vector<std::vector<float>> distances;
    distances.resize(num_query_pts);
    for (auto &el : distances) {
      el.resize(k);
    }
    nanoflann::KNNResultSet<float> resultSet(k);
    float out_dists[k];
    size_t neighbors[k];

    for (tensorflow::int64 query_pt_idx = 0; query_pt_idx < num_query_pts;
         ++query_pt_idx) {

      const float query_pt[3] = {query_pts_tensor(query_pt_idx, 0),
                                 query_pts_tensor(query_pt_idx, 1),
                                 query_pts_tensor(query_pt_idx, 2)};

      resultSet.init(neighbors, out_dists);

      index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

      for (tensorflow::int64 neighbor_idx = 0; neighbor_idx < k; ++neighbor_idx) {
        eig_dists(query_pt_idx, neighbor_idx) = out_dists[neighbor_idx];
        eig_indices(query_pt_idx, neighbor_idx) =
            static_cast<int32_t>(neighbors[neighbor_idx]);
      }
    }
  }
};

template struct KNNFunctor<CPUDevice, float>;

} // namespace functor
} // namespace tensorflow
