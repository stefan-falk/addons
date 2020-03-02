#define EIGEN_USE_THREADS

#include "tensorflow_addons/custom_ops/cci/cc/kernels/hungarian.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_addons/custom_ops/cci/cc/kernels/munkres.h"

namespace tensorflow {
namespace addons {

template <typename Device, typename T>
class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& costs_tensor = context->input(0);
    auto costs = costs_tensor.tensor<T, 3>();

    Tensor* assignments_tensor = nullptr;
    std::vector<int64> cost_shape;
    for (int i = 0; i < costs_tensor.shape().dims(); ++i) {
      cost_shape.push_back(costs_tensor.shape().dim_size(i));
    }
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({cost_shape[0], cost_shape[1]}),
                                &assignments_tensor));
    auto assignments_output = assignments_tensor->matrix<int>();

    const int batch_size = cost_shape[0];
    auto shard = [&costs, &cost_shape, &assignments_output](int64 start,
                                                            int64 limit) {
      for (int n = start; n < limit; ++n) {
        Matrix<T> matrix(cost_shape[1], cost_shape[2]);
        for (int i = 0; i < cost_shape[1]; ++i) {
          for (int j = 0; j < cost_shape[2]; ++j) {
            matrix(i, j) = costs(n, i, j);
          }
        }
        Munkres<T> munk = Munkres<T>();
        munk.solve(matrix);

        for (int i = 0; i < cost_shape[1]; ++i) {
          bool assigned = false;
          for (int j = 0; j < cost_shape[2]; ++j) {
            if (matrix(i, j) == 0) {
              assigned = true;
              assignments_output(n, i) = j;
            }
            if (!assigned) assignments_output(n, i) = -1;
          }
        }
      }
    };

    const int64 single_cost = cost_shape[1] * cost_shape[2];

    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(batch_size, single_cost, shard);
  }
};

#define REGISTER_HUNGARIAN_KERNELS(T)                              \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Hungarian").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      HungarianOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_HUNGARIAN_KERNELS);
TF_CALL_float(REGISTER_HUNGARIAN_KERNELS);
TF_CALL_double(REGISTER_HUNGARIAN_KERNELS);
#undef REGISTER_HUNGARIAN_KERNELS
}  // namespace addons
}  // namespace tensorflow