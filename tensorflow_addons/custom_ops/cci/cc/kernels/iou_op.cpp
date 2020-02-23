#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "iou_op.h"


namespace tensorflow {
namespace addons{
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class PolygonIouOp : public OpKernel {
 public:
  explicit PolygonIouOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& polygon1_tensor = context->input(0);
    const Tensor& polygon2_tensor = context->input(1);
    const int batch_size = polygon1_tensor.shape().dim_size(0);
    const int edge_count1 = polygon1_tensor.shape().dim_size(1);
    const int edge_count2 = polygon2_tensor.shape().dim_size(1);
    OP_REQUIRES(context,
                (polygon1_tensor.shape().dims() == 3) &&
                    polygon1_tensor.shape().dim_size(2) == 2,
                errors::InvalidArgument("Input shapes must be [N,E,2]"));
    OP_REQUIRES(context,
                (polygon2_tensor.shape().dims() == 3) &&
                    polygon2_tensor.shape().dim_size(2) == 2,
                errors::InvalidArgument("Input shapes must be [N,E,2]"));
    Tensor* iou_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size}), &iou_tensor));
    auto polygon1_t = polygon1_tensor.tensor<T, 3>();
    auto polygon2_t = polygon2_tensor.tensor<T, 3>();
    auto iou_f = iou_tensor->flat<T>();
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    thread_pool->ParallelFor(batch_size, edge_count1 * edge_count2 * 1000,
                             [&](const int64 start, const int64 end) {
                               for (int64 i = start; i < end; i++) {
                                 auto iou = iou_poly(polygon1_t, polygon2_t, i);
                                 iou_f(i) = iou;
                               }
                             });
  }

 private:
  T iou_poly(typename TTypes<T, 3>::ConstTensor p,
             typename TTypes<T, 3>::ConstTensor q, int64 index) {
    int64 n1 = p.dimension(1);
    int64 n2 = q.dimension(1);
    Point<T> ps1[n1], ps2[n2];
    for (int64 i = 0; i < n1; i++) {
      ps1[i].x = p(index, i, 0);
      ps1[i].y = p(index, i, 1);
    }
    for (int64 i = 0; i < n2; i++) {
      ps2[i].x = q(index, i, 0);
      ps2[i].y = q(index, i, 1);
    }

    T inter_area = intersectArea(ps1, n1, ps2, n2);
    T union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    T iou = inter_area / union_area;
    return abs(iou);
  }
};

#define REGISTER_POLYGON_IOU_KERNELS(T)                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("PolygonIou").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PolygonIouOp<CPUDevice, T>);
TF_CALL_float(REGISTER_POLYGON_IOU_KERNELS);
TF_CALL_double(REGISTER_POLYGON_IOU_KERNELS);
#undef REGISTER_POLYGON_IOU_KERNELS
}
}  // namespace tensorflow
