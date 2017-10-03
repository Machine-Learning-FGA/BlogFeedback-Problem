import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from termcolor import cprint
from sklearn.feature_selection import RFECV


print('Lendo dados', end='')
#carregando dados

data = pd.read_csv('blogdata/train.csv')

print(data)

x_train = np.array(data.iloc[0:10000, 0:1])
y_train = np.array(data.iloc[0:10000,-1])

cprint(' Done', 'green')


print('Criando dados de test', end='')
#dados de test

data_test = pd.read_csv('blogdata/test1.csv')

x_test = np.array(data_test.iloc[:, 0:1])
y_test = np.array(data_test.iloc[:,-1])


cprint(' Done', 'green')


#Iniciando modelo
print('Iniciando modelo', end='')

#model = Lasso(fit_intercept=True, normalize=True)
model = Ridge()
cprint(' Done', 'green')

#treinando modelo

print('Treinando modelo', end='')


rfe = RFECV(model)
x_train = rfe.fit_transform(x_train,y_train)

model.fit(x_train,y_train)

print(cross_val_score(model,x_test,y_test))
cprint(' Done', 'green')
