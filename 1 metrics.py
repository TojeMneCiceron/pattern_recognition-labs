import math
import pandas as pd
import numpy as np

def Euclid(xi, xj, p):
    sum = 0
    for k in range(p):
        sum += (xi[k]-xj[k])**2
    return math.sqrt(sum)

def Hamming(xi, xj, p):
    sum = 0
    for k in range(p):
        sum += abs(xi[k]-xj[k])
    return sum

def Manhattan(xi, xj, p):
    max = 0
    for k in range(p):
        d = abs(xi[k]-xj[k])
        if d > max:
            max = d
    return max

def Jaccard(a, b, n):
    sum_ab = 0
    sum_a = 0
    sum_b = 0
    for k in range(n):
        sum_ab += a[k]*b[k]
        sum_a += a[k]
        sum_b += b[k]
    return 1 - sum_ab / (sum_a + sum_b - sum_ab)

def Cosine(a, b, n):
    sum_ab = 0
    sum_a = 0
    sum_b = 0
    for k in range(n):
        sum_ab += a[k]*b[k]
        sum_a += a[k]**2
        sum_b += b[k]**2
    return np.arccos(sum_ab / (math.sqrt(sum_a) * math.sqrt(sum_b)))

def Metrics(train, test):
    p = len(test[0]) - 1
    for i in range(len(test)):
        min_e = (Euclid(test[i], train[0], p), 0)
        min_h = (Hamming(test[i], train[0], p), 0)
        min_m = (Manhattan(test[i], train[0], p), 0)
        min_j = (Jaccard(test[i], train[0], p), 0)
        min_c = (Cosine(test[i], train[0], p), 0)
        for k in range(len(train)):
            e = Euclid(test[i], train[k], p)
            h = Hamming(test[i], train[k], p)
            m = Manhattan(test[i], train[k], p)
            j = Jaccard(test[i], train[k], p)
            c = Cosine(test[i], train[k], p)
            if e < min_e[0]:
                min_e = (e, k)
            if h < min_h[0]:
                min_h = (h, k)
            if m < min_m[0]:
                min_m = (m, k)
            if j < min_j[0]:
                min_j = (j, k)
            if c < min_c[0]:
                min_c = (c, k)  
        print(test[i]) 
        print("Euclid: " + str(train[min_e[1]]))
        print("Hamming: " + str(train[min_h[1]]))
        print("Manhattan: " + str(train[min_m[1]]))
        print("Jaccard: " + str(train[min_j[1]]))
        print("Cosine: " + str(train[min_c[1]]))
        print()

data = pd.read_csv(r"C:\Users\Пользователь\python\lab1\iris.csv")
data = np.random.permutation(data)

n_training = math.ceil(len(data)*0.9)

train = data[:n_training]
print("Train:")
print(train)
test = data[n_training:]
print("Test:")
print(test)
print()
Metrics(train, test)
input()