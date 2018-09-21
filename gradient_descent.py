import numpy as np
import matplotlib.pyplot as plt

class Data:
	def __init__(self, x=[], y=[]):
		self.x = np.array(x)
		self.y = np.array(y)
		if not x or not y or len(x) is not len(y):
			raise ValueError('x and y muse be same size and non-empty')

	def plot(self, best_fit=False, title='', xlabel='', ylabel=''):
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		plt.scatter(self.x, self.y)
		if best_fit:
			t0, t1 = self.grad_des()
			print(type(t1))
			xs = np.array(np.arange(min(self.x), max(self.x), 0.0001))
			line = (t0 + (t1*xs)).astype(np.int)
			plt.plot(self.x, [t0 + (t1*i) for i in self.x])
			print(t0, t1)
		plt.show()

	hypothesis = lambda self, t0, t1,: t0 + (t1*self.x)

	def cost(self, t0=0, t1=1):
		
		m = len(self.x)
		sum_error = sum(np.power(self.hypothesis(t0, t1) - self.y, 2))
		J = (1.0/(2*m))*sum_error
		return J
	def gradient(self, t0, t1):
		m = len(self.x)
		derive_t0 = (1/m) * sum((self.hypothesis(t0, t1) - self.y))
		derive_t1 = (1/m) * sum((self.hypothesis(t0, t1) - self.y) * self.x)
		return derive_t0, derive_t1
	def grad_des(self, learning_rate=0.001):
		theta0 = 0
		theta1 = 0
		alpha = learning_rate # step per iter
		diff = 99999999999999
		while(diff > 1e-40):
			d_t0, d_t1 = self.gradient(theta0, theta1)
			temp0 = theta0 - (alpha*d_t0)
			temp1 = theta1 - (alpha*d_t1)
			diff = np.power((theta0 + theta1)/2 - (temp0 + temp1)/2, 2)

			theta0 = temp0
			theta1 = temp1
			print(diff)
		return theta0, theta1





	def model(self):
		pass
		# return t0, t1


d = Data(x=[1, 2, 3, 4, 5, ], y=[2, 4, 6, 8, 10])
d.plot(best_fit=True)













