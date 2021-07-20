# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
from typing import Union, Dict, Hashable, List

import tensorflow as tf

from tensorflow_addons.utils import types


class AccumulationGradientTransformer:
    _accu_gradients: Union[Dict[Hashable, tf.Variable], None] = None

    def __init__(
        self,
        optimizer: types.Optimizer,
        accu_steps: types.TensorLike,
        trainable_variables,
    ):
        self.optimizer = optimizer
        self.accu_steps = accu_steps
        self.step = tf.Variable(0, dtype=tf.int64, name="ga_step")
        self._accu_gradients: Union[List[tf.Variable], None] = None
        policy = tf.keras.mixed_precision.global_policy()
        self.variable_dtype = policy.variable_dtype
        self._accu_gradients = {
            var.ref(): self.optimizer.add_slot(var, "ga") for var in trainable_variables
        }

    def __call__(self, grads_and_vars, *args, **kwargs):

        variables = [var for (_, var) in grads_and_vars]
        accu_gradients = self._accu_gradients
        step_inc_op = self.step.assign_add(1, read_value=False)

        with tf.control_dependencies([step_inc_op]):
            can_apply = tf.cast(
                self.step % self.accu_steps == 0, dtype=self.variable_dtype
            )
            accumulate = tf.cast(
                self.step % (self.accu_steps + 1) != 0, dtype=self.variable_dtype
            )

        accum_ops = list()
        for grad, var in grads_and_vars:

            # Get the accumulated gradient
            grad_accum = accu_gradients[var.ref()] * accumulate

            if isinstance(grad, tf.IndexedSlices):
                # Not sure why e.g. the Embedding layer requires an additional dimension here
                grad_indices = grad.indices[..., None] if len(grad.indices.shape) < 2 else grad.indices
                added = tf.IndexedSlices(
                    values=grad.values
                    + tf.gather_nd(grad_accum, grad_indices),
                    indices=grad.indices,
                    dense_shape=grad.dense_shape,
                )
                accu_op = accu_gradients[var.ref()].scatter_update(added)
            else:
                accu_op = accu_gradients[var.ref()].assign(
                    grad + grad_accum, read_value=False
                )

            accum_ops.append(accu_op)

        iter_dec_op = self.optimizer.iterations.assign_add(
            -1 * tf.cast(can_apply, dtype=self.optimizer.iterations.dtype),
            read_value=False,
        )

        with tf.control_dependencies(accum_ops + [iter_dec_op]):
            gradients = [accu_gradients[var.ref()] * can_apply for var in variables]
            return list(zip(gradients, variables))


def GradientAccumulator(
    optimizer: types.Optimizer, accu_steps: int = 2, trainable_variables=None
) -> types.Optimizer:
    if trainable_variables is None:
        trainable_variables = list()

    if isinstance(optimizer, str):
        optimizer = tf.keras.optimizers.get(optimizer)

    optimizer.gradient_transformers.append(
        AccumulationGradientTransformer(
            optimizer=optimizer,
            accu_steps=accu_steps,
            trainable_variables=trainable_variables,
        )
    )

    return optimizer
