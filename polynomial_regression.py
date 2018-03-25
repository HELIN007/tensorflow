# coding: utf-8
'''
曲线拟合
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


N = 100
x_train = np.linspace(-3, 3, N, dtype=np.float32)
y_train = np.sin(x_train) + np.random.uniform(0, 1, N)


def model_fn(features, labels, mode):
    W = tf.get_variable('Wight_1', [1], tf.float32)
    W_1 = tf.get_variable('Wight_2', [1], tf.float32)
    W_2 = tf.get_variable('Wight_3', [1], tf.float32)
    b = tf.get_variable('bias', [1], tf.float32)
    # 拟合2、3次方
    add = W * features['x'] + W_1 * tf.pow(features['x'], 2) + W_2 * tf.pow(features['x'], 3)
    y = tf.add(add, b, name='output')
    predictions = {
        'y': y
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.mean_squared_error(labels, predictions=y)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)


config = tf.estimator.RunConfig(model_dir='polynomial_model',
                                keep_checkpoint_max=1,
                                save_checkpoints_steps=10)
polynomial_regression = tf.estimator.Estimator(model_fn=model_fn,
                                               config=config)


input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train},
                                              y_train, batch_size=10, num_epochs=None, shuffle=True)


tensor_to_log = {"y": "output"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)


polynomial_regression.train(input_fn=input_fn, steps=100, hooks=[logging_hook])

new_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, None, batch_size=10, shuffle=False)

predictions = list(polynomial_regression.predict(input_fn=new_input_fn))
y_pre = []
for i in predictions:
    y_pre.append(i['y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_train, y_train)
ax.plot(x_train, y_pre, color='r')
plt.show()
