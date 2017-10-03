import numpy as np 
import pandas as pd


data = pd.read_csv('blogdata/train.csv')

x = data.iloc[:,0:281]
print(x.head())

x = x.as_matrix()

print(np.corrcoef(x))
