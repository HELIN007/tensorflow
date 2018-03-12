# -*- coding=utf-8 -*-
# Python3.6.4
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]


def my_input_fn(train_file, is_shuffle=False, repeat_count=1):
    dataset = tf.data.TFRecordDataset(train_file)  # filename得是个list

    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature((), tf.int64),
            # 这一步features返回的shape要对应下面的my_features
            # 比如这里是(1, 4)，下面读取就要是parsed['features'][0][0]->[0][3]
            # 如果是(4,)，下面读取就得是parsed['features'][0]->[4]
            'features': tf.FixedLenFeature((4,), tf.float32),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        my_features = {
            'SepalLength': parsed['features'][0],
            'SepalWidth': parsed['features'][1],
            'PetalLength': parsed['features'][2],
            'PetalWidth': parsed['features'][3]
        }
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

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[10, 10],  # Two layers, each with 10 neurons
    n_classes=3,  # 3个特征值
    model_dir='iris_model_csv_tfrecords')  # 存储模型的地址

train_file = ['train_iris.tfrecords']
test_file = ['test_iris.tfrecords']

classifier.train(input_fn=lambda: my_input_fn(train_file, is_shuffle=True, repeat_count=100))

# 用test_file数据来计算模型
# 模型能够算出accuracy, average_loss, loss, global_step
evaluate_result = classifier.evaluate(input_fn=lambda: my_input_fn(test_file, is_shuffle=True, repeat_count=4))
print("Evaluation results: ")
for key in evaluate_result:
    print("    {} was: {}".format(key, evaluate_result[key]))

predict_results = classifier.predict(input_fn=lambda: my_input_fn(test_file, is_shuffle=False, repeat_count=1))
print("Predictions on test file: ")
print([prediction["class_ids"][0] for prediction in predict_results])


# 现在测试输入的数据
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa


def new_input_fn():
    def decode(x):
        x = tf.split(x, 4)  # 需要分成4(自己的数据里的features个数)个features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # 在测试里，没有label


# 显示出预测的结果
predict_results = classifier.predict(input_fn=new_input_fn)
print("Predictions on memory")
for idx, prediction in enumerate(predict_results):
    type = prediction["class_ids"][0]  # Get the predicted class (index)
    if type == 0:
        print("I think {} is Iris Sentosa".format(prediction_input[idx]))
    elif type == 1:
        print("I think {} is Iris Versicolor".format(prediction_input[idx]))
    else:
        print("I think {} is Iris Virginica".format(prediction_input[idx]))
