licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_cci_ops.so",
    srcs = [
        "cc/kernels/hungarian.cc",
        "cc/kernels/hungarian.h",
        "cc/kernels/iou_op.cc",
        "cc/kernels/iou_op.h",
        "cc/kernels/matrix.cc",
        "cc/kernels/matrix.h",
        "cc/kernels/munkres.cc",
        "cc/kernels/munkres.h",
        "cc/ops/hungarian.cc",
        "cc/ops/iou_op.cc",
    ],
)

py_library(
    name = "iou_ops",
    srcs = [
        "hungarian.py",
        "iou_ops.py",
    ],
    data = [
        "//tensorflow_addons/custom_ops/cci:_cci_ops.so",
        "//tensorflow_addons/utils",
    ],
)
