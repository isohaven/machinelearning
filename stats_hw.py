import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('movies.txt', header='infer')
cell_text = []
table = plt.table(cellText=df.values, colLabels=df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
# table.scale(1.5, 1.5)
plt.show()
# print(df.columns)