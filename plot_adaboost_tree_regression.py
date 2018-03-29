# -*- coding=utf-8 -*-
# Python3.6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# 制作数据集
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

regression_1 = DecisionTreeRegressor(max_depth=5)
regression_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                                 n_estimators=300, random_state=rng)
regression_1.fit(X, y)
regression_2.fit(X, y)

y_1 = regression_1.predict(X)
y_2 = regression_2.predict(X)

plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
