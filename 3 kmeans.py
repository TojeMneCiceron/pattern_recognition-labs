import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data=pd.read_csv(r"C:\Users\Пользователь\python\lab1\iris.csv")
X = data.iloc[:,:-1].to_numpy()
X = np.random.permutation(X)


def RecountZ(X):
    X = X[:,:-1]
    a = np.mean(X, axis=0)
    # print('-----------------------')
    # print(X)
    # print('---------')
    # print(a)
    # print('-----------------------')
    return a

def Kmeans(X, z, k):
    i = 0
    while(True):
        i += 1
        new_z = []
        for i in range(k):
            new_z.append(RecountZ(X[X[:,-1] == i]))

        # print('---------------------------------')
        a = np.array(new_z)
        b = np.array(z)
        # print(a)
        # print(b)
        sum = np.sum(a-b)
        # print(sum)
        # print('---------------------------------')

        if sum < 0.00000001:
            print('----iter:', i)
            print('----new_z:')
            print(*(new_z), sep = '\n')
            return X
        else:
            z = new_z
            Cluster(X[:,:-1], new_z)

def Cluster(X, z):
    cluster_n = []

    for x in X:
        d = np.min(np.sqrt(np.sum(np.square(x-z),axis=1)))
        cluster_n.append([np.argmin(np.sqrt(np.sum(np.square(x-z),axis=1)))])

    res = np.append(X, np.array(cluster_n), axis=1)
    return res

k = 3

z = X[:k]
print('----z:')
print(z)

X = Cluster(X, z)

X = Kmeans(X, z, k)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=X[:,-1])
plt.show()