import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()

class Model():
    def __init__(self):
        self.learning_rate = 0.01
    
    def fit(self, X, y):
        self.w = np.random.rand(len(X[0]))
        self.b = 0
        while True:
            update_count = 0
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                if (yi * (np.dot(self.w, xi) + self.b)) <= 0:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi
                    update_count += 1
            if update_count == 0:
                break
        print("Finished!")
    
    def ans(self):
        return (self.w, self.b)

X_train = data.data[:100][:,0:2]
y_train = data.target[:100]
y_train[y_train == 0] = -1

model = Model()
model.fit(X_train, y_train)

w, b = model.ans()

print("w: {}, b: {}".format(w, b))

x0 = np.arange(4,7,0.1)
x1 = - (w[0] * x0 + b) / w[1]
plt.plot(x0, x1)

plt.scatter(X_train[:50,0], X_train[:50,1], label='0')
plt.scatter(X_train[50:,0], X_train[50:,1], label='1')
plt.legend()
plt.show()