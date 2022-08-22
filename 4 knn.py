import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\Пользователь\python\lab1\iris.csv")
x = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
k = 3

def KNN(x_train, x_test, y_train, k):
    res = []
    for x in x_test:
        d = np.sqrt(np.sum(np.square(x_train - x), axis = 1))
        nearest = y_train[d.argsort()[0:k]]
        unique = np.unique(nearest)
        res.append(unique[0])
    return res
        
predicted = KNN(x_train, x_test, y_train, k)

res = np.column_stack((y_test, predicted))   
print(*(res), sep = '\n')