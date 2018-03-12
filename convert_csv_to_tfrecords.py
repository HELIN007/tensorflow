# -*- coding=utf-8 -*-
# Python 3.6
'''
本代码只是将csv文件转换成了tfrecords文件，
一般是将train和test的文件做成tfrecords文件，predict文件直接pandas读取再输入
要注意解析的时候的问题。
'''

import tensorflow as tf
import pandas as pd


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    # 本来传入的就是个list，所以value不用加[]了
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def creat_tfrecords(filename, to_name):
    # 先对csv文件进行处理，pandas处理
    # 第一行不当做属性，然后跳过第一行，防止读取出错！
    csv = pd.read_csv(filename, header=None, skiprows=1).values
    print(csv)
    with tf.python_io.TFRecordWriter(to_name) as writer:
        for row in csv:  # 一行一行的读取
            # 每行的最后一位为label，前面的为features
            # 这里的features和label要看数据格式
            # 需要int的话下面就是_int64_feature()，需要float的话就是_float_feature()
            features, label = row[:-1], int(row[-1])
            features = [float(f) for f in features]
            # 解析的时候要参考制作的时候的features, label的类型
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature(label),
                        'features': _float_feature(features)
                    }
                )
            )
            writer.write(example.SerializeToString())
        print('制作 %s 文件完成！' % to_name)


def main():
    train_filename = 'iris_training.csv'
    test_filename = 'iris_test.csv'
    creat_tfrecords(train_filename, 'train_iris.tfrecords')
    creat_tfrecords(test_filename, 'test_iris.tfrecords')


if __name__ == '__main__':
    main()
