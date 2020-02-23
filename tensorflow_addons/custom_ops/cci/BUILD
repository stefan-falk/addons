licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_cci_ops.so",
    srcs = [
        "cc/kernels/iou_op.cc",
        "cc/kernels/iou_op.h",
    ]
)
