# -*- coding=utf-8 -*-
# Python3.6

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=80, centers=2, random_state=2)
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# plt.cm.Paired输出两个相近颜色
# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y)

# 获取当前图层，继续画图
ax = plt.gca()
# 获取最大最小坐标值
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# 网格点
XX, YY = np.meshgrid(xx, yy)

# 将坐标点一一对应，n*2数据
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)

CS = ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                alpha=0.5, linestyles=['--', '-', '--'])
# 给每条直线上标注信息（等高线的写法）
fmt = {}
strs = ['label 0', 'hyperplane', 'label 1']
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax.clabel(CS, inline=True, fmt=fmt, fontsize=10)

# 标注出支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           c='none', s=100, linewidth=1, edgecolors='k')

plt.show()
