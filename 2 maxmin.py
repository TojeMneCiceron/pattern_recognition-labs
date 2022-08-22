import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data=pd.read_csv(r"C:\Users\Пользователь\python\lab1\iris.csv")
X = data.iloc[:,:-1].to_numpy()

def CheckZ(L, Z):
    d = 0
    for z in Z:
        d += np.sum(np.sqrt(np.sum(np.square(Z-z),axis=1)))/(len(Z)-1)
    d /= len(Z)

    return L > (d/2)

def Maxmin(X, z):
    while(True):
        L = 0
        new_z = X[0]

        for x in X:
            min = np.min(np.sqrt(np.sum(np.square(x-z),axis=1)))
            if min > L:
                L = min
                new_z = x
                
        check = CheckZ(L, z)
        # print(check)
        if (check):
            z.append(new_z)
        else:
            return z

def Cluster(X, z):
    cluster_n = []

    for x in X:
        d = np.min(np.sqrt(np.sum(np.square(x-z),axis=1)))
        if (d==0):
            cluster_n.append([len(z)+1])
        else:
            cluster_n.append([np.argmin(np.sqrt(np.sum(np.square(x-z),axis=1)))])

    res = np.append(X, np.array(cluster_n), axis=1)
    return res

z = [X[random.randrange(0,len(X))]]
z.append(X[np.argmax(np.sqrt(np.sum(np.square(X-z),axis=1)))])

z = Maxmin(X, z)
print(len(z))
print(z)

res = Cluster(X, z)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:,0], res[:,1], res[:,2], c=res[:,-1])
plt.show()