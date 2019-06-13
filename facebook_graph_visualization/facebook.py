import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

data = []
with open('facebook.txt', 'r') as f:
	for line in f:
		data.append(line.replace('\n', '').split(','))

plt.xlabel('age')
plt.ylabel('count')
plt.title('facebook user age distribution')
del data[1][0]
classes = ['18-22', '23-27', '28+']
bar1 = plt.bar(np.arange(0, 3, 1)-.2, [78,49,21], color='blue', label='yes', width=.3)
bar2 = plt.bar(np.arange(0, 3, 1)+.1, [4,21,46], color='red', label='no', width=.3, align='center')
plt.xticks(np.arange(0, 3, 1), ['18-22', '23-27', '28+'])
plt.legend(loc='upper right', title='facebook user?')
plt.show()

# d = [1, 2, 3, 4, 5]
# print(np.std(d, ddof=0))

