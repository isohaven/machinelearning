import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

xs = np.arange(1, 51) # xs [1-50]
ys = np.random.random((1, 50))[0] # ys randoms [0-1]
X = np.column_stack((xs, ys)) # X is featureset [xs, ys]
# print(X)
plt.scatter(X[:,0], X[:, 1], s=100)
plt.show()
