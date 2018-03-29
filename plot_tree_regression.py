# -*- coding=utf-8 -*-
# Python3.6

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import pydotplus


# 创建随机数据
rng = np.random.RandomState(1)
# 80*1，按列从小到大排序
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# 增加噪声
y[::5] += 3 * (0.5 - rng.rand(16))

regression_1 = DecisionTreeRegressor(max_depth=2)
regression_2 = DecisionTreeRegressor(max_depth=5)
regression_3 = DecisionTreeRegressor(max_depth=15)
regression_1.fit(X, y)
regression_2.fit(X, y)
regression_3.fit(X, y)
dot_data = tree.export_graphviz(regression_1, out_file=None, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('sin.pdf')

# 预测
X_test = np.arange(0.0, 5.0, 0.01).reshape(-1, 1)
y_1 = regression_1.predict(X_test)
y_2 = regression_2.predict(X_test)
y_3 = regression_3.predict(X_test)


plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen",
         label="max_depth=5", linewidth=2)
plt.plot(X_test, y_3, color="red",
         label="max_depth=15", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
