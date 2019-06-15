import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

fig, ax  = plt.subplots()

def draw_circle(x=0, y=0, r=1):
    circle = plt.Circle( (x, y), r, alpha=0.35)
    ax.add_artist(circle)
def neighbors(point, X, distance):
    friends = [] 
    for x in X:
        if( np.linalg.norm(point-x) <= distance):
            friends.append(x)
    return friends
def kernel(point1, point2):
    distance = np.linalg.norm(point1-point2)
    return k(p1- p2)
xs = np.arange(1, 51) # xs [1-50]
ys = 50 * np.random.random((1, 50))[0] # ys randoms [0-1]
X = np.column_stack((xs, ys)) # X is featureset [xs, ys]
# print(X)
draw_circle(15, 15, 7)
plt.scatter(X[:,0], X[:, 1], s=100)
plt.scatter(15, 15, c='b')
print(neighbors( [15, 15], X, 7))
plt.show()

