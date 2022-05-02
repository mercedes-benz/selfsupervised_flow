// Copyright 2022 MBition GmbH
// SPDX-License-Identifier: MIT

//----------------------------------------------------------------------
// Thanks to our colleague Christoph for the following PointsToVoxelOp!
//----------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <unordered_map>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

// x, y, z
using Coordinate = Eigen::Array3i;

auto coord_cmp = [](const Coordinate& a, const Coordinate& b) -> bool { return (a == b).all(); };

namespace std {
template <>
struct hash<Coordinate> {
  std::size_t operator()(const Coordinate& c) const {
    using std::hash;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:
    return ((hash<int32_t>()(c[0]) ^ (hash<int32_t>()(c[1]) << 1)) >> 1) ^
           (hash<int32_t>()(c[2]) << 1);
  }
};

}  // namespace std

namespace tensorflow {

class LinearSectioningAxis {
 public:
  using Scalar = float;

  LinearSectioningAxis() : n_(1), min_(0.0), max_(1.0), diff_(1.0), n_flt_(1.0) {}
  LinearSectioningAxis(Scalar min, Scalar max, std::int32_t n)
      : n_(n), min_(min), max_(max), diff_(max - min), n_flt_(static_cast<Scalar>(n_)) {}

  bool integrate(Scalar value, std::int32_t& i) const {
    Scalar zeroed = value - min_;
    if (zeroed < 0.0)
      return false;
    i = static_cast<std::int32_t>(zeroed / diff_ * n_flt_);
    return i < n_;
  }

  std::int32_t n() const { return n_; }

 private:
  std::int32_t n_;
  Scalar min_;
  Scalar max_;
  Scalar diff_;
  Scalar n_flt_;
};

class PointsToVoxelOp : public OpKernel {
 public:
  // [x y z reflectivity]
  static constexpr std::uint8_t INPUT_FEATURES_PER_POINT = 4;
  // [(x y z)/(L W H) r | (x_rel_mean y_rel_mean z_rel_mean)/(voxel size) r_rel_mean | (x_rel_center y_rel_center z_rel_center)/(voxel size)]
  static constexpr std::uint8_t OUTPUT_FEATURES_PER_POINT = 4 + 4 + 3;
  // [(x_mean y_mean z_mean)/(L W H) r_mean | (x_mean_rel_center y_mean_rel_center z_mean_rel_center)/(voxel size) | (x_center y_center z_center)/(L W H) | nbr_valid_points/max_nbr_points]
  static constexpr std::uint8_t OUTPUT_FEATURES_PER_VOXEL = 4 + 3 + 3 + 1;

  using MapType = std::unordered_map<Coordinate,
                                     std::vector<std::uint32_t>,
                                     std::hash<Coordinate>,
                                     decltype(coord_cmp)>;

  explicit PointsToVoxelOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input voxel extent/voxel number -> configure voxel grid
    auto tensor_voxel_extent = context->input(1).flat<float>();
    auto tensor_voxel_number = context->input(2).flat<int32_t>();
    // axes order is: X - Y - Z
    for (uint8_t i = 0; i < 3; ++i) {
      voxel_extent_min_[i] = tensor_voxel_extent.data()[i];
      voxel_extent_max_[i] = tensor_voxel_extent.data()[3 + i];
      axes_[i] =
          LinearSectioningAxis{tensor_voxel_extent.data()[i], tensor_voxel_extent.data()[3 + i],
                               tensor_voxel_number.data()[i]};
    }

    auto voxel_number = Eigen::Map<const Eigen::Array3i>(tensor_voxel_number.data());
    voxel_cell_size_ = (voxel_extent_max_ - voxel_extent_min_) / voxel_number.cast<float>();
    volume_ = voxel_extent_max_ - voxel_extent_min_;

    // max_points_per_voxel
    const auto max_points_per_voxel = context->input(3).flat<std::int32_t>()(0);

    // grab input point cloud tensor
    const Tensor& tensor_point_cloud = context->input(0);
    // [batch, n points, xyzr]
    auto input = tensor_point_cloud.tensor<float, 3>();
    const auto batch_size = input.dimension(0);

    // vector of hash_maps, one per point cloud
    std::vector<MapType> point_maps{};
    point_maps.reserve(batch_size);
    // vector of point_counters, one per point cloud
    std::vector<std::int64_t> point_counters(batch_size, 0);

    const std::size_t point_cloud_data_size = input.dimension(1) * INPUT_FEATURES_PER_POINT;

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      // put all points into map [Coordinate -> Vector of points]
      // count number of points actually located within voxel grid
      const float* const point_cloud_begin = input.data() + batch * point_cloud_data_size;
      point_maps.emplace_back(sort_into_voxels(
          point_cloud_begin, point_cloud_begin + point_cloud_data_size, &point_counters[batch]));
    }

    const std::int64_t num_occupied_voxels =
        std::accumulate(point_maps.cbegin(), point_maps.cend(), std::int64_t{0},
                        [](std::int64_t a, const auto& point_map) {
                          return a + static_cast<std::int64_t>(point_map.size());
                        });
    const std::int64_t point_counter =
        std::accumulate<std::vector<std::int64_t>::const_iterator, std::int64_t>(
            std::cbegin(point_counters), std::cend(point_counters), std::int64_t{0});

    // *******************************************************
    // Create output tensors: v_features, p_features, v_coordinates, v_counters,
    // features, mapping
    // *******************************************************
    Tensor* ot_v_features = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape{{num_occupied_voxels, OUTPUT_FEATURES_PER_VOXEL}},
            &ot_v_features));
    auto ot_v_features_eigen = ot_v_features->tensor<float, 2>();

    Tensor* ot_p_features = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape{{num_occupied_voxels, max_points_per_voxel, OUTPUT_FEATURES_PER_POINT}},
            &ot_p_features));
    auto ot_p_features_eigen = ot_p_features->tensor<float, 3>();

    Tensor* ot_v_coordinates = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{{num_occupied_voxels, 4}},
                                                     &ot_v_coordinates));
    auto ot_v_coordinates_eigen = ot_v_coordinates->tensor<int32_t, 2>();

    Tensor* ot_v_counters = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, TensorShape{{num_occupied_voxels}}, &ot_v_counters));
    auto ot_v_counters_eigen = ot_v_counters->tensor<int32_t, 1>();

    Tensor* ot_features = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       4, TensorShape{{point_counter, OUTPUT_FEATURES_PER_POINT}}, &ot_features));
    auto ot_features_eigen = ot_features->tensor<float, 2>();

    Tensor* ot_mapped_voxels = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(5, TensorShape{{point_counter}}, &ot_mapped_voxels));
    auto ot_mapped_voxels_eigen = ot_mapped_voxels->tensor<int32_t, 1>();

    Tensor* ot_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(6, TensorShape{{point_counter, 2}}, &ot_indices));
    auto ot_indices_eigen = ot_indices->tensor<int32, 2>();

    // *******************************************************
    // Fill outputs
    // *******************************************************

    // set unfilled feature space to NaN beforehand
    ot_p_features_eigen.setConstant(NAN);

    std::int32_t voxel_idx = 0;
    std::int64_t point_idx = 0;
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      const float* const point_cloud_begin =
          input.data() + batch * (input.dimension(1) * INPUT_FEATURES_PER_POINT);

      const auto& coord_map = point_maps[batch];
      for (auto it = coord_map.cbegin(); it != coord_map.cend(); ++it, ++voxel_idx) {
        // counter
        ot_v_counters_eigen(voxel_idx) =
            std::min<std::int32_t>(it->second.size(), max_points_per_voxel);
        // coordinates [batch, x, y, z]
        ot_v_coordinates_eigen(voxel_idx, 0) = batch;
        for (uint8_t c = 0; c < 3; ++c)
          ot_v_coordinates_eigen(voxel_idx, c + 1) = it->first[c];

        // features
        const Eigen::Vector4f mean_point = getMeanInVoxel(point_cloud_begin, it->second);
        const Eigen::Array3f voxel_center = getVoxelCenterPoint(it->first);

        // [(x_mean y_mean z_mean)/(L W H) r_mean | (x_mean_rel_center y_mean_rel_center z_mean_rel_center)/(voxel size) | (x_center y_center z_center)/(L W H) | nbr_valid_points/max_nbr_points]
        for (uint8_t c = 0; c < 3; ++c){
          ot_v_features_eigen(voxel_idx, c) = mean_point[c] / volume_[c];
          ot_v_features_eigen(voxel_idx, c + 4) = (mean_point[c] - voxel_center[c]) / voxel_cell_size_[c];
          ot_v_features_eigen(voxel_idx, c + 7) = voxel_center[c] / volume_[c];
        }
        ot_v_features_eigen(voxel_idx, 3) = mean_point[3];
        ot_v_features_eigen(voxel_idx, 10) = static_cast<float>(it->second.size()) / max_points_per_voxel;

        for (std::size_t p = 0; p < it->second.size(); ++p, ++point_idx) {
          std::size_t p_index = it->second[p];

          const Eigen::Vector4f centered_point_mean_point =
              Eigen::Map<const Eigen::Vector4f>(point_cloud_begin +
                                                p_index * INPUT_FEATURES_PER_POINT) -
              mean_point;
          const Eigen::Array3f centered_point_voxel =
              Eigen::Map<const Eigen::Array3f>(point_cloud_begin +
                                               p_index * INPUT_FEATURES_PER_POINT) -
              voxel_center;

          ot_mapped_voxels_eigen(point_idx) = voxel_idx;
          ot_indices_eigen(point_idx, 0) = batch;
          ot_indices_eigen(point_idx, 1) = p_index;

          // features for every input point
          for (uint8_t c = 0; c < 3; ++c) {
            ot_features_eigen(point_idx, c) =
                point_cloud_begin[p_index * INPUT_FEATURES_PER_POINT + c];
            ot_features_eigen(point_idx, c + 4) = centered_point_mean_point[c];
            ot_features_eigen(point_idx, c + 7) = centered_point_voxel[c];
          }
          ot_features_eigen(point_idx, 3) =
              point_cloud_begin[p_index * INPUT_FEATURES_PER_POINT + 3];

          // add as voxel feature if still < max_points_per_voxel
          if (p < (std::size_t) max_points_per_voxel) {
            for (uint8_t c = 0; c < 3; ++c) {
              // [(x y z)/(L W H) r | (x_rel_mean y_rel_mean z_rel_mean)/(voxel size) r_rel_mean | (x_rel_center y_rel_center z_rel_center)/(voxel size)]
              ot_p_features_eigen(voxel_idx, p, c) =
                  point_cloud_begin[p_index * INPUT_FEATURES_PER_POINT + c] / volume_[c];
              ot_p_features_eigen(voxel_idx, p, c + 4) = centered_point_mean_point[c] / voxel_cell_size_[c];
              ot_p_features_eigen(voxel_idx, p, c + 8) = centered_point_voxel[c] / voxel_cell_size_[c];
            }
            ot_p_features_eigen(voxel_idx, p, 3) =
                point_cloud_begin[p_index * INPUT_FEATURES_PER_POINT + 3];
            ot_p_features_eigen(voxel_idx, p, 7) = centered_point_mean_point[3];
          }
        }
      }
    }
  }

 private:
  bool makeCoord(const float* point_cloud_begin, Coordinate& coord) const {
    for (uint8_t i = 0; i < 3; ++i) {
      if (!axes_[i].integrate(static_cast<LinearSectioningAxis::Scalar>(point_cloud_begin[i]),
                              coord[i]))
        return false;
    }
    return true;
  }

  bool isPaddedValue(const float* point_begin) const {
    return std::any_of(point_begin, point_begin + INPUT_FEATURES_PER_POINT,
                       [](float v) { return std::isnan(v); });
  }

  Eigen::Vector4f getMeanInVoxel(const float* point_cloud_begin,
                                 const std::vector<std::uint32_t>& point_indices) const {
    return std::accumulate(point_indices.cbegin(), point_indices.cend(),
                           Eigen::Vector4f{Eigen::Vector4f::Zero()},
                           [point_cloud_begin](const Eigen::Vector4f& sum,
                                               std::uint32_t index) -> Eigen::Vector4f {
                             return sum + Eigen::Map<const Eigen::Vector4f>(
                                              point_cloud_begin + index * INPUT_FEATURES_PER_POINT);
                           }) /
           static_cast<float>(point_indices.size());
  }

  Eigen::Array3f getVoxelCenterPoint(const Coordinate& coord) const {
    return (coord.cast<float>() + 0.5) * voxel_cell_size_ + voxel_extent_min_;
  }

  MapType sort_into_voxels(const float* point_cloud_begin,
                           const float* point_cloud_end,
                           std::int64_t* point_counter) {
    const MapType::size_type n = 100;
    MapType coords(n, std::hash<Coordinate>{}, coord_cmp);
    std::uint32_t point_index = 0;  //< index in original input point cloud
    for (; point_cloud_begin != point_cloud_end;
         point_cloud_begin += INPUT_FEATURES_PER_POINT, ++point_index) {
      // end of point cloud reached (padded with zeros)
      if (isPaddedValue(point_cloud_begin))
        continue;
        // shuffled point clouds are padded inbetween points, so go through the whole point cloud actually
        // also, it isn't wasting that much computation, as pcls are almost equal size in homogenous data
        // return coords;

      Coordinate coord{};
      if (!makeCoord(point_cloud_begin, coord))  //< not within voxel extent
        continue;

      const auto it = coords.find(coord);
      if (it == coords.end()) {
        coords.emplace(coord, std::vector<std::uint32_t>{point_index});
      } else {
        it->second.push_back(point_index);
      }
      ++(*point_counter);
    }
    return coords;
  }

  //! hold voxel grid extent and voxel size/number
  std::array<LinearSectioningAxis, 3> axes_;
  //! Voxel grid definition as Eigen Arrays
  Eigen::Array3f voxel_cell_size_;
  Eigen::Array3f volume_;
  Eigen::Array3f voxel_extent_min_;
  Eigen::Array3f voxel_extent_max_;
};

Status ResizeShapeFn(::tensorflow::shape_inference::InferenceContext* c) {
  using namespace ::tensorflow::shape_inference;
  ShapeHandle shape;
  DimensionHandle dim;
  // point cloud
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shape));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(shape, 2), PointsToVoxelOp::INPUT_FEATURES_PER_POINT, &dim));
  // voxel grid extent
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape));
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(shape, 0), 6, &dim));
  // voxel number
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape));
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(shape, 0), 3, &dim));
  // max points per voxel
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &shape));

  // voxel features
  c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
    PointsToVoxelOp::OUTPUT_FEATURES_PER_VOXEL}));
  const auto tensor_points = c->input_tensor(3);
  int32_t num_points = InferenceContext::kUnknownDim;
  if (tensor_points != nullptr) {
    num_points = (*tensor_points).flat<int32_t>()(0);
  }
  // point features
  c->set_output(1, c->MakeShape({InferenceContext::kUnknownDim, num_points,
                                 PointsToVoxelOp::OUTPUT_FEATURES_PER_POINT}));
  // voxel coordinates
  c->set_output(2, c->MakeShape({InferenceContext::kUnknownDim, 4}));
  // voxel counters
  c->set_output(3, c->MakeShape({InferenceContext::kUnknownDim}));
  // features
  c->set_output(
      4, c->MakeShape({InferenceContext::kUnknownDim, PointsToVoxelOp::OUTPUT_FEATURES_PER_POINT}));
  // mapped voxels
  c->set_output(5, c->MakeShape({InferenceContext::kUnknownDim}));
  // mapped indices
  c->set_output(6, c->MakeShape({InferenceContext::kUnknownDim, 2}));
  return Status::OK();
}

REGISTER_OP("PointsToVoxel")
    // batched (padded) point clouds [B x P_max X 4]. Padded with NaNs
    .Input("point_cloud: float32")
    // specify extent of voxel grid. Shape = (6, ). [min_x, min_y, min_z, max_x,
    // max_y, max_z]
    .Input("voxel_grid_extent: float32")
    // specify number of voxels. Shape = (3, ). [#x, #y, #z]
    .Input("voxel_number: int32")
    // specify maximum number of points per voxel
    .Input("max_points_per_voxel: int32")
    // [K x OUTPUT_FEATURES_PER_VOXEL]
    .Output("voxel_features: float32")
    // [K x Points_per_voxel X OUTPUT_FEATURES_PER_POINT]
    .Output("point_features: float32")
    // [K x 4] Coordinates: batch, x, y, z
    .Output("voxel_coordinates: int32")
    // [K] Number of valid points per voxel <= max number points
    .Output("voxel_counters: int32")
    // # features for every input point falling into the grid, but might be larger
    // # than max_points_per_voxel
    .Output("features: float32")
    .Output("mapped_voxels: int32")
    .Output("mapped_indices: int32")
    .SetShapeFn(ResizeShapeFn)
    .Doc(R"doc(
point_cloud: batched (padded) point clouds [B x P_max X 4]. Padded with NaNs
voxel_grid_extent: specify extent of voxel grid. Shape = (6, ). [min_x, min_y, min_z, max_x, max_y, max_z]
voxel_number: specify number of voxels. Shape = (3, ). [#x, #y, #z]
max_points_per_voxel: specify maximum number of points per voxel

# Outputs:
voxel_features: [K x OUTPUT_FEATURES_PER_VOXEL]
point_features: [K x Points_per_voxel X OUTPUT_FEATURES_PER_POINT]
voxel_coordinates: [K x 4] Coordinates: batch, x, y, z
voxel_counters: [K] Number of valid points per voxel <= max number points

# OUTPUT_FEATURES_PER_VOXEL: 11
    (x_mean y_mean z_mean)/(L W H) r_mean
    (x_mean_rel_center y_mean_rel_center z_mean_rel_center)/(voxel size)
    (x_center y_center z_center)/(L W H)
    nbr_valid_points/max_nbr_points
# OUTPUT_FEATURES_PER_POINT: 11
    (x y z)/(L W H) r
    (x_rel_mean y_rel_mean z_rel_mean)/(voxel size) r_rel_mean
    (x_rel_center y_rel_center z_rel_center)/(voxel size)

Old docu:
This op takes a point cloud [batch size x nbr points x 4] and a voxel grid
as input. The grid is specified by its overall extent
(min_x, min_y, min_z, max_x, max_y, max_z) and the number of voxels along each
dimension (#x, #y, #z). Additionally, a maxmimum number of points per voxel is
considered. According to the VoxelNet paper, the K non-empty voxels
(over the complete batch) are returned.
The 8 voxel features are
(x_mean y_mean z_mean r_mean | x_center y_center z_center | nbr valid points)
[K x 8].
The 11 point features are
(x y z r | x_rel_mean y_rel_mean z_rel_mean r_rel_mean |
x_rel_center y_rel_center z_rel_center)
[K x max points per voxel x 11].
The 4 voxel coordinates are (batch idx, x, y, z) [K x 4].
The number of valid and used points in each voxel is given as additional tensor
[K]. The 7 (= 11 - rel_center - r_rel_mean) point features as simple list over
all points [nbr points x 7].
Map from the list of points to the voxel indices in range(K) [nbr points].
Batch and point indices in Voxel [nbr points x 2].
)doc"
    );

REGISTER_KERNEL_BUILDER(Name("PointsToVoxel").Device(DEVICE_CPU), PointsToVoxelOp);

}  // namespace tensorflow
