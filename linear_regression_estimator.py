# -*- coding: utf-8 -*-
'''
线性回归预测模型，estimator处理
'''

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode):
    W = tf.get_variable('Wight', [1], dtype=tf.float32)
    b = tf.get_variable('bias', [1], dtype=tf.float32)
    # 转化x的格式以及方便predictions输出
    x = tf.cast(features['x'], tf.float32, name='x_input')
    # 预测值y
    y = tf.add(W * x, b, name='y_pre')
    # 一维化为batch_size维，方便predictions输出
    W_pre = tf.add(W, x * 0, name='W_pre')
    b_pre = tf.add(b, x * 0, name='b_pre')
    # 预测的内容
    predictions = {
        'x': x,
        "y": y,
        'W': W_pre,
        'b': b_pre
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 两个loss都对，但学习率需要设置的不一样
    loss = tf.losses.mean_squared_error(labels=labels, predictions=y)
    # loss = tf.reduce_sum(tf.square(y - labels))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss
    )


config = tf.estimator.RunConfig(
    model_dir='linear_model',  # 存放位置
    keep_checkpoint_max=1,  # 最多保存的模型数量
    # save_summary_steps=5,  # tensorboard
    save_checkpoints_steps=10,  # 每隔多少步保存一个checkpoint
)

linear_classfier = tf.estimator.Estimator(
    model_fn=model_fn,
    config=config
)

# tensors_to_log里面可以放好几个想输出的东西
tensors_to_log = {
    "y": "y_pre",
    "x": "x_input",
    # "W": "W_pre",
    # "b": "b_pre"
}
# 放置钩子，输出想要的内容，每every_n_iter次打印一次结果
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100
)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([2., 3., 4., 5.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

linear_classfier.train(input_fn=input_fn, steps=100, hooks=[logging_hook])

# 预测自己的数据
x_pre = np.array([2., 3, 4, 5, 6, 7, 8])
new_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_pre}, None, batch_size=4, shuffle=False)
predictions = linear_classfier.predict(input_fn=new_input_fn)
for i in predictions:
    print(i)
