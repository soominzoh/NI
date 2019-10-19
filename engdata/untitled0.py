import pandas as pd
data = pd.read_csv(r'C:\Users\Z\Desktop\NI\engdata\ni_w_scores.csv')

data.corr()

import matplotlib.pyplot as plt 
import seaborn as sns    

plt.figure(figsize=(15,15))
sns.heatmap(data = data.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Greens')