import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
    ])
# plt.scatter(X[:,0], X[:,1], s=100)
# plt.show()
colors = 5*['g', 'r', 'c', 'b', 'k']

class K_Means:
    def __init__(self, k=2, tol=0.01, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def fit(self, data):
        self.centroids = {}
        # first k centroids are chosen arbitraraly
        # in this case the first k features are the initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # contains (centroid, fetures) pairs
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                # list of distances from each centroid from the feature
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # centroid changes to mean of features
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            # see if converged
            optimized = True
            for c in self.centroids:
                origanal_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-origanal_centroid) / origanal_centroid*100.0) > self.tol:
                    print('centroid moved: ', np.sum((current_centroid-origanal_centroid) / origanal_centroid*100.0))
                    optimized = False
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
clf = K_Means(k=2)
clf.fit(X)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
        marker = 'o', color='k',s=50)
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=70,linewidth=3)


unknowns = np.array([
    [2, 4],
    [5, 4],
    [8, 8],
    ])
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', c=colors[classification], s=70)

plt.show()
