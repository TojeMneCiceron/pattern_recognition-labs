import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\Пользователь\python\lab1\iris.csv")
x = data.iloc[:,:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()

y[y=='setosa']=-1
y[y=='versicolor']=1
y[y=='virginica']=1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

def Hyperplane(X, w, b):
    return X.dot(w) + b

def Margin(X, y, w, b):
    return y * Hyperplane(X, w, b)
 
def SVM(X, y, C, lr=1e-3, epochs=500):
    n, d = X.shape
    w = np.random.randn(d)
    b = 0

    for e in range(epochs):
        margin = Margin(X, y, w, b)

        i_misclssd = np.where(margin < 1)[0]
        d_w = w - C * y[i_misclssd].dot(X[i_misclssd])
        w = w - lr * d_w

        d_b = - C * np.sum(y[i_misclssd])
        b -= lr * d_b

    return w, b
 
def Predict(X, w, b):
    res = np.sign(Hyperplane(X, w, b))
    return res
 
def SVM_predict(x_test, y_test, w, b):    
    predicted = []
    for xt in x_test:
        predicted.append(int(Predict(xt, w, b)))

    res = np.column_stack((y_test, predicted)) 
    print(*(res), sep = '\n')  
    return predicted

C = 1
w, b = SVM(x_train, y_train, C, 0.001, 500)

y_pred = SVM_predict(x_test, y_test, w, b)