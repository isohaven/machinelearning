import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

xs = np.arange(1, 51)
ys = np.random.random((1, 50))[0]
print(len(xs))
print(len(ys))
print(xs)
print(ys)
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
    ])
# plt.scatter(X[:,0], X[:, 1], s=100)
# plt.show()
