# -*- coding=utf-8 -*-
# Python 3.6

import pathlib
from PIL import Image
import tensorflow as tf


def resize_picture():
    p = pathlib.Path('.') / 'train'

    is_dirs = [x for x in p.iterdir() if x.is_dir()]
    # print(is_dirs)

    dogs_path = is_dirs[0]
    dogs_files = [x for x in dogs_path.iterdir() if x.is_file()]
    cats_path = is_dirs[1]
    cats_files = [x for x in cats_path.iterdir() if x.is_file()]

    for i in range(len(dogs_files)):
        name = str(dogs_files[i]).split('/')[-1].replace('dog.', 'dog')
        im = Image.open(dogs_files[i])
        im = im.resize((128, 128))
        path = 'train_dogs/' + name
        if pathlib.Path(path).exists():
            print('{0}已存在'.format(name))
        else:
            print('正在转化第%d张dog图片...' % i)
            im.save(path, 'JPEG')

    for i in range(len(cats_files)):
        name = str(cats_files[i]).split('/')[-1].replace('cat.', 'cat')
        im = Image.open(cats_files[i])
        im = im.resize((128, 128))
        path = 'train_cats/' + name
        if pathlib.Path(path).exists():
            print('{0}已存在'.format(name))
        else:
            print('正在转化第%d张cat图片...' % i)
            im.save(path, 'JPEG')
    print('转化图片完成！')


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def creat_tfrecords(filename):
    # 人为设定，相当于labels标签
    classes = {'train_cats', 'train_dogs'}
    file_path = pathlib.Path('.').cwd() / filename
    if pathlib.Path(file_path).exists():
        print('{0}已存在'.format(filename))
    else:
        with tf.python_io.TFRecordWriter(filename) as writer:
            # 循环已有的labels标签
            print('正在写入中...')
            for index, name in enumerate(classes):
                # labels的路径
                cwd = pathlib.Path('.').cwd() / name
                # 该labels下的所有图片
                img_files = [x for x in cwd.iterdir() if x.is_file()]
                # 循环每张图片
                for i in range(len(img_files)):
                    img = Image.open(img_files[i])
                    img_raw = img.tobytes()  # 将图片转化为二进制
                    # example对象对labels和image数据进行封装
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "label": int64_feature(index),
                                "img_raw": bytes_feature(img_raw)
                            }
                        )
                    )
                    # 序列化为字符串
                    writer.write(example.SerializeToString())
        print('生成%s文件成功！' % filename)


def read_tfrecords(filename):
    # 根据文件名生成一个队列，此处的filename得是个list
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 取出包含image和label的feature对象
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    imgs = tf.decode_raw(features['img_raw'], tf.uint8)
    imgs = tf.reshape(imgs, [128, 128, 3])
    imgs = tf.cast(imgs, tf.float32)
    labels = tf.cast(features['label'], tf.int32)
    print('images数据: {0} \nlabels数据：{1}'.format(imgs, labels))
    print('--------------------------------------------------')
    if 0:
        # 其实就是查看转化好的images数据和labels数据，以便自己验证
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 启动多线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(51):  # 不知道如何得到Tensor的长度
                image, label = sess.run([imgs, labels])
                if 0:
                    # 要保存图片的话
                    img = Image.fromarray(image, 'RGB')
                    img.save()
                print(image.shape, label)
            # 记得关线程
            coord.request_stop()
            coord.join(threads)
    # 返回的是两个Tensor张量
    return imgs, labels


def use_tfrecords(img, label):
    # 队列读取
    # num_threads:使用多个线程读取文件
    # batch_size:表示进行一次批处理的tensors数量
    # capacity:队列中的最大的元素数，这个参数一定要比min_after_dequeue参数的值大，
    # 并且决定了可以进行预处理操作元素的最大值。
    # min_after_dequeue:当一次出列操作完成后,队列中元素的最小数量，用于定义元素的混合级别，要小于capacity
    # 定义了随机取样的缓冲区大小，此参数越大表示更大级别的混合但是会导致启动更加缓慢，并且会占用更多的内存
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    num_threads=2,
                                                    batch_size=20,
                                                    capacity=40,
                                                    min_after_dequeue=30)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            # print(img_batch.shape, label_batch.shape)
            val, l = sess.run([img_batch, label_batch])
            print(val.shape, l)
        coord.request_stop()
        coord.join(threads)


def dataset_read_tfrecords(filename):
    dataset = tf.data.TFRecordDataset(filename)  # filename得是个list

    def parser(record):
        keys_to_features = {
            'img_raw': tf.FixedLenFeature((), tf.string, default_value=""),
            'label': tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64))
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['img_raw'], tf.uint8)
        image = tf.reshape(image, [128, 128, 3])
        label = tf.cast(parsed['label'], tf.int64)
        return image, label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(10)
    dataset = dataset.repeat(10)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    print(features, labels)
    # 验证数据
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     a, b = sess.run([features, labels])
    #     print(a.shape, b)
    #     coord.request_stop()
    #     coord.join(threads)
    return features, labels


def main():
    filename = ['cats_dogs.tfrecords']
    # resize_picture()
    creat_tfrecords(filename[0])
    img, label = read_tfrecords(filename)
    # use_tfrecords(img, label)
    dataset_read_tfrecords(filename)


if __name__ == '__main__':
    main()
