import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

fig, ax  = plt.subplots()

norm = lambda x1, x2: np.linalg.norm(x1-x2)
def draw_circle(x=0, y=0, r=1):
    circle = plt.Circle( (x, y), r, alpha=0.35)
    ax.add_artist(circle)
def neighbors(point, X, distance):
    friends = [] 
    for x in X:
        if( norm(point, x) <= distance):
            friends.append(x)
    return friends
def kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)



