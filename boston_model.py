# -*- coding=utf-8 -*-
# Python3.6.4
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

feature_names = [
    'CRIM',
    'ZN',
    'INDUS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'TAX',
    'PTRATIO'
]

if 0:
    # 制作tf文件
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        # 本来传入的就是个list，所以value不用加[]了
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def creat_tfrecords(filename, to_name):
        # 先对csv文件进行处理，pandas处理
        csv = pd.read_csv(filename, header=None, skiprows=1).values
        with tf.python_io.TFRecordWriter(to_name) as writer:
            for row in csv:  # 一行一行的读取
                # 每行的最后一位为label，前面的为features
                features, label = row[:-1], row[-1]
                features = [float(f) for f in features]
                # 解析的时候要参考制作的时候的features, label的类型
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': _float_feature([label]),
                            'features': _float_feature(features)
                        }
                    )
                )
                writer.write(example.SerializeToString())
            print('%s has been converted!' % to_name)

    train_filename = 'boston_train.csv'
    test_filename = 'boston_test.csv'
    creat_tfrecords(train_filename, 'train_boston.tfrecords')
    creat_tfrecords(test_filename, 'test_boston.tfrecords')
    print('------------------------------------------------')


def my_input_fn(train_file, is_shuffle=False, repeat_count=1):
    dataset = tf.data.TFRecordDataset(train_file)  # filename得是个list

    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature((), tf.float32),
            # 这一步features返回的shape要对应下面的my_features
            # 比如这里是(1, 4)，下面读取就要是parsed['features'][0][0]->[0][3]
            # 如果是(4,)，下面读取就得是parsed['features'][0]->[4]
            'features': tf.FixedLenFeature((9,), tf.float32),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        my_features = {}
        # dict的key和feature_names要一一对应
        for idx, key in enumerate(feature_names):
            my_features[key] = parsed['features'][idx]
        return my_features, parsed['label']

    dataset = dataset.map(parser)
    # 打乱顺序，buffer_size设置成一个大于数据集中样本数量的值来确保其充分打乱
    if is_shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    # 打包成多少个进行输出
    dataset = dataset.batch(10)
    # 指定要遍历几遍整个数据集
    dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


# 要使得自己设置的feature_names和解析出来的features的key一致
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

regressor = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[10, 10],  # Two layers, each with 10 neurons
    model_dir='boston_model_tfrecords')  # 存储模型的地址

train_file = ['train_boston.tfrecords']
test_file = ['test_boston.tfrecords']

regressor.train(input_fn=lambda: my_input_fn(train_file, is_shuffle=True, repeat_count=10))

# 用test_file数据来计算模型
# 模型能够算出accuracy, average_loss, loss, global_step
evaluate_result = regressor.evaluate(input_fn=lambda: my_input_fn(test_file, is_shuffle=False, repeat_count=4))
print("Evaluation results: ")
for key in evaluate_result:
    print("    {} was: {}".format(key, evaluate_result[key]))

predict_results = regressor.predict(input_fn=lambda: my_input_fn(test_file, is_shuffle=False, repeat_count=1))
print("Predictions on test file: ")
print([float('%.1f' % prediction["predictions"]) for prediction in predict_results])


prediction_input = pd.read_csv('boston_predict.csv', header=None, skiprows=1).values


def new_input_fn():
    def decode(x):
        x = tf.split(x, 9)  # 需要分成9(自己的数据里的features个数)个features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # 在测试里，没有label


# 显示出预测的结果
predict_results = regressor.predict(input_fn=new_input_fn)
print("Predictions on memory")
for idx, prediction in enumerate(predict_results):
    print(idx, float('%.1f' % prediction["predictions"]))
