import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from termcolor import cprint
from sklearn.externals import joblib


print('Lendo dados para Treinamento')
data_train = pd.read_csv('blog_dataset/train.csv', header=None)
x_train = np.array(data_train.iloc[0:20000, 0:280])
y_train = np.array(data_train.iloc[0:20000, -1])
cprint('Dados lidos com sucesso!', 'green')

print('Criando dados para Teste')
data_test = pd.read_csv('blog_dataset/test.csv', header=None)
x_test = np.array(data_test.iloc[:, 0:280])
y_test = np.array(data_test.iloc[:, -1])
cprint('Dados para teste criados com sucesso!', 'green')

vt = VarianceThreshold(threshold=(.80 * (1 - .80)))
x_train_vt = vt.fit_transform(x_train, y_train)
x_test_vt = vt.fit_transform(x_test, y_test)
print(x_train.shape)
print(x_train_vt.shape)

'''
clf = ExtraTreesClassifier()
train_clf = clf.fit(x_train,y_train)
test_clf = clf.fit(x_test,y_test)
train_clf.feature_importances_
test_clf.feature_importances_

model_train = SelectFromModel(train_clf, prefit=True)
model_test = SelectFromModel(test_clf, prefit=True)

x_train_clf = model_train.transform(x_train)
x_test_clf = model_test.transform(x_test)
print(x_train.shape)
print(x_train_clf.shape)
'''


print('Treinando Modelo')
regr = DecisionTreeRegressor()

regr.fit(x_train_vt,y_train)

print("Obtendo Resultados")
y = np.array(y_test)
x = np.array(x_test)

print(cross_val_score(regr, x_test_vt, y_test, scoring='explained_variance'))

joblib.dump(regr, 'decisionTreeRegressor.pkl')
