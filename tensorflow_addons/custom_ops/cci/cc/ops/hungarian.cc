#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

REGISTER_OP("Hungarian")
    .Attr("T: {int32,float,double}")
    .Input("cost_matrix: T")
    .Output("assignments: int32")

    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(0);

      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      const int32 input_rank = c->Rank(input);
      std::vector<shape_inference::DimensionHandle> dims;

      for (int i = 0; i < input_rank - 1; ++i) {
        dims.emplace_back(c->Dim(input, i));
      }

      c->set_output(0, c->MakeShape(dims));

      return Status::OK();
    });
}  // namespace addons
}  // namespace tensorflow