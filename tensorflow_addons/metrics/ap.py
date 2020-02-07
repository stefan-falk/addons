# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements AP."""

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
def _common_iou(b1, b2, mode='iou'):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['iou', 'ciou', 'diou', 'giou'], decided to calculate IoU or CIoU or DIoU or GIoU.
    Returns:
        IoU loss float `Tensor`.
    """
    b1 = tf.convert_to_tensor(b1)
    if not b1.dtype.is_floating:
        b1 = tf.cast(b1, tf.float32)
    b2 = tf.cast(b2, b1.dtype)

    def _inner():
        zero = tf.convert_to_tensor(0., b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        if mode == 'iou':
            return iou

        elif mode in ['ciou', 'diou']:
            enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
            enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
            enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
            enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

            b1_center = tf.stack([(b1_ymin + b1_ymax) / 2,
                                  (b1_xmin + b1_xmax) / 2])
            b2_center = tf.stack([(b2_ymin + b2_ymax) / 2,
                                  (b2_xmin + b2_xmax) / 2])
            euclidean = tf.linalg.norm(b2_center - b1_center)
            diag_length = tf.linalg.norm(
                [enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin])
            diou = iou - (euclidean**2) / (diag_length**2)
            if mode == 'ciou':
                v = _get_v(b1_height, b1_width, b2_height, b2_width)
                alpha = tf.math.divide_no_nan(v, ((1 - iou) + v))
                return diou - alpha * v

            return diou
        elif mode == 'giou':
            enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
            enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
            enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
            enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
            enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
            enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
            enclose_area = enclose_width * enclose_height
            giou = iou - tf.math.divide_no_nan(
                (enclose_area - union_area), enclose_area)
            return giou
        else:
            raise ValueError(
                "Value of mode should be one of ['iou','giou','ciou','diou']")

    return tf.squeeze(_inner())


def iou(b1, b2):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
    Returns:
        IoU loss float `Tensor`.
    """
    return _common_iou(b1, b2, 'iou')

@tf.keras.utils.register_keras_serializable(package="Addons")
class ObjectDetectionAP(tf.keras.metrics.Metric):
    def __init__(
        self,
        iou_threshold=0.5,
        prediction_threshold=0.0,
        name="mean_average_precision",
        dtype=tf.float32,
    ):
        super().__init__(name=name)

        if iou_threshold is not None or prediction_threshold is not None:
            if not isinstance(iou_threshold, float):
                raise TypeError("The value of iou_threshold should be a python float")
            if iou_threshold > 1.0 or iou_threshold < 0.0:
                raise ValueError("Iou_threshold should be between 0 and 1")
            if not isinstance(prediction_threshold, float):
                raise TypeError("The value of prediction_threshold should be a python float")
            if prediction_threshold > 1.0 or prediction_threshold < 0.0:
                raise ValueError("Prediction_threshold should be between 0 and 1")

        self.iou_threshold = iou_threshold
        self.prediction_threshold = prediction_threshold

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=None, initializer="zeros", dtype=self.dtype
            )

        self.ap = _zero_wt_init("true_positives")


    # y_pred [[ymin,xmin,ymax,xmax,prediction]] [N,5]
    # y_true [[ymin,xmin,ymax,xmax]] [N,4]
    def update_state(self, y_true, y_pred, sample_weight=None):
        size=tf.shape(y_pred)[0]
        true_positives=tf.TensorArray(self.dtype,size+1,clear_after_read=False,tensor_array_name='true_positives')
        false_positives=tf.TensorArray(self.dtype,size+1,clear_after_read=False,tensor_array_name='false_positives')
        true_positives=true_positives.write(0,0)
        false_positives=false_positives.write(0,0)
        unique_arr=tf.TensorArray(self.dtype,size,dynamic_size=True,clear_after_read=False,tensor_array_name='unique')
        
        arr_index=0

        def _contain(array,value):
            is_contain=False
            for i in tf.range(array.size()):
                if array.read(i)==value:
                    is_contain=True
            return is_contain

        def _add_true(true_positives,false_positives,index):
            true_positives=true_positives.write(index+1,true_positives.read(index)+1)
            false_positives=false_positives.write(index+1,false_positives.read(index))

        def _add_false(true_positives,false_positives,index):
            true_positives=true_positives.write(index+1,true_positives.read(index))
            false_positives=false_positives.write(index+1,false_positives.read(index)+1)

        for i in tf.range(size):
            max_iou=tf.reduce_max(iou(y_true[i,:4],y_pred[:,:4]))
            if max_iou>=self.iou_threshold and y_pred[i,4]>=self.prediction_threshold:
                unique_arr=unique_arr.write(arr_index,i)
                if _contain(unique_arr,tf.cast(i,self.dtype)):
                    _add_false(true_positives,false_positives,arr_index)
                else:
                    _add_true(true_positives,false_positives,arr_index)
            else:
                _add_false(true_positives,false_positives,arr_index)
            arr_index+=1
        
        true_positives=true_positives.stack('true_positives')
        false_positives=false_positives.stack('false_positives')
        sorted_index=tf.argsort(y_pred[:,4], direction='DESCENDING')
        y_pred_sorted=tf.gather(y_pred,sorted_index)
        true_positives=tf.gather(true_positives,sorted_index)
        false_positives=tf.gather(false_positives,sorted_index)
        precision=true_positives/(true_positives+false_positives)
        recall=true_positives/tf.cast(tf.shape(y_true)[0],self.dtype)
        ap=tf.cast(0,self.dtype)
        for i in tf.range(size):
            ap+=(recall[i+1]-recall[i])*precision[i+1]
        self.ap.assign(ap)
        
    def result(self):
        return self.ap

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "iou_threshold": self.iou_threshold,
            "prediction_threshold": self.prediction_threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        self.ap.assign(0)

