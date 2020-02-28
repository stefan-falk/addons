import tensorflow as tf
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons.utils import types

_activation_so = LazySO("custom_ops/activations/_activation_ops.so")


def polygon_iou(
    area: types.FloatTensorLike, predicts: types.FloatTensorLike
) -> tf.Tensor:
    """Polygon IoU function.

    Args:
        area: Interest area, Tensorlike float or double, shape must be [E,2] or [N,E,2].
        predicts: Predict results, Tensorlike float or double, shape must be [N,E,2], has the same type as `area`.
    Returns:
        Tensor, shape [N], output of polygon iou. Has the same type as `area`.
    """
    return _activation_so.ops._polygon_iou(area, predicts)
