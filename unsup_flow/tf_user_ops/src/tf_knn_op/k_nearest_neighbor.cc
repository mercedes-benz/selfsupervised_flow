// Copyright 2022 MBition GmbH
// SPDX-License-Identifier: MIT

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace shape_inference {
Status KNNShapeFn(::tensorflow::shape_inference::InferenceContext *c) {
  using namespace ::tensorflow::shape_inference;
  ShapeHandle shape_ref;
  ShapeHandle shape_query;
  DimensionHandle dim;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shape_ref));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &shape_query));
  const auto dim_or_const_ref = DimensionOrConstant(c->Dim(shape_ref, 1));
  const auto dim_or_const_query = DimensionOrConstant(c->Dim(shape_query, 1));
  if (InferenceContext::ValueKnown(dim_or_const_ref) && InferenceContext::ValueKnown(dim_or_const_query)
      && InferenceContext::Value(dim_or_const_ref) != InferenceContext::Value(dim_or_const_query))
  {
    return Status(error::Code::FAILED_PRECONDITION,
                  "Dimension of ref and query points does not match.");
  }

  std::int64_t n = InferenceContext::kUnknownDim;
  const auto dim_or_const_query_n = DimensionOrConstant(c->Dim(shape_query, 0));
  if (InferenceContext::ValueKnown(dim_or_const_query_n))
    n = InferenceContext::Value(dim_or_const_query_n);


    tensorflow::int64 k = -1;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

  c->set_output(0, c->MakeShape({n,k}));
  c->set_output(1, c->MakeShape({n,k}));

  return Status::OK();
}
}  // namespace shape_inference

REGISTER_OP("KNearestNeighbor")
    .Attr("T: realnumbertype = DT_FLOAT")
    .Attr("k: int")
    .Input("ref: T")
    .Input("query: T")
    .Output("indices: int32")
    .Output("distances: T")
    .SetShapeFn(shape_inference::KNNShapeFn);

}  // namespace tensorflow
