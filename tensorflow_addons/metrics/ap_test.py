import tensorflow as tf
from tensorflow_addons.metrics import ObjectDetectionAP
from tensorflow_addons.utils import test_utils

class APTest(tf.test.TestCase):
    def test_basic(self):
        preds = tf.constant([[0, 0, 1, 1, 0.7], [0, 0, 1, 1, 0.3], [0, 0, 1, 1, 0.3]])
        actuals = tf.constant([[0, 0, 1, 0.8], [0, 0, 0.5, 1], [0, 0, 0.1, 0.1]])
        expect_result = 0.231
        ap_metric=ObjectDetectionAP()
        ap_metric.update_state(actuals,preds)
        actual_result = ap_metric.result()
        self.assertAllClose(actual_result,expect_result)

if __name__=='__main__':
    tf.test.main()
