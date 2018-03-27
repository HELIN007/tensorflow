# -*- coding=utf-8 -*-
# Python3.6

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap

plt.rcParams.update({'figure.autolayout': True})

np.random.seed(1)

fig = plt.figure(figsize=(4, 6))
ax1 = fig.add_subplot(211)
# 自己设计一些数据，均值方差正态分布
x1 = np.random.normal(50, 6, 200)
y1 = np.random.normal(5, 0.5, 200)

x2 = np.random.normal(30, 6, 200)
y2 = np.random.normal(4, 0.5, 200)

x3 = np.random.normal(45, 6, 200)
y3 = np.random.normal(2.5, 0.5, 200)

x_val = np.concatenate((x1, x2, x3))
y_val = np.concatenate((y1, y2, y3))

# 归一化数据
x_diff = max(x_val) - min(x_val)
y_diff = max(y_val) - min(y_val)
x_normalized = x_val / x_diff
y_normalized = y_val / y_diff
xy_normalized = list(zip(x_normalized, y_normalized))
# 每个数据的标签
labels = [1] * 200 + [2] * 200 + [3] * 200

# knn算法
clf = neighbors.KNeighborsClassifier(12)
clf.fit(xy_normalized, labels)

# 测试两个数据
# 返回测试数据附近的5个邻居，不显示距离
nearests = clf.kneighbors([(50 / x_diff, 5 / y_diff), (30 / x_diff, 3 / y_diff)], 5, False)
print(nearests)
# 预测这两数据的种类
prediction = clf.predict([[50 / x_diff, 5 / y_diff], [30 / x_diff, 3 / y_diff]])
print(prediction)
# 预测这两数据的属于每种类的概率
prediction_proba = clf.predict_proba([(50 / x_diff, 5 / y_diff), (30 / x_diff, 3 / y_diff)])
print(prediction_proba)


# 测试数据，数据一样
x1_test = np.random.normal(50, 6, 100)
y1_test = np.random.normal(5, 0.5, 100)

x2_test = np.random.normal(30, 6, 100)
y2_test = np.random.normal(4, 0.5, 100)

x3_test = np.random.normal(45, 6, 100)
y3_test = np.random.normal(2.5, 0.5, 100)

xy_test_normalized = list(zip(np.concatenate((x1_test, x2_test, x3_test)) / x_diff,
                              np.concatenate((y1_test, y2_test, y3_test)) / y_diff))

# 测试标签
labels_test = [1] * 100 + [2] * 100 + [3] * 100
# 返回平均准确率
score = clf.score(xy_test_normalized, labels_test)
print(score)

# 测试不同的k值时的准确率
clf1 = neighbors.KNeighborsClassifier(6)
clf1.fit(xy_normalized, labels)
score1 = clf1.score(xy_test_normalized, labels_test)
print(score1)


# 划区域图，上色
xx, yy = np.meshgrid(np.arange(1, 70.1, 0.1), np.arange(1, 7.01, 0.01))
xx_normalized = xx / x_diff
yy_normalized = yy / y_diff
coords = np.c_[xx_normalized.ravel(), yy_normalized.ravel()]
Z = clf.predict(coords)
Z = Z.reshape(xx.shape)

light_rgb = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
ax1.pcolormesh(xx, yy, Z, cmap=light_rgb)
ax1.scatter(x1, y1, c='b', marker='s', s=20, alpha=0.8)
ax1.scatter(x2, y2, c='r', marker='^', s=20, alpha=0.8)
ax1.scatter(x3, y3, c='g', s=20, alpha=0.8)
ax1.axis((10, 70, 1, 7))

# 测试30个k，10折交叉验证，选取最佳k值
k_range = range(1, 31)
k_scores = []
for k in k_range:
    clf = neighbors.KNeighborsClassifier(k)
    scores = cross_val_score(clf, xy_normalized, labels, cv=10, scoring='accuracy')  # for classification
    k_scores.append(scores.mean())


ax2 = fig.add_subplot(212)
ax2.plot(k_range, k_scores)
ax2.set_xlabel('Value of K for KNN')
ax2.set_ylabel('Cross-Validated Accuracy')
plt.show()
