import numpy as np
from BasicML import visdata


class Regression:

    def __init__(self, x, y, w=[], d=1):
        self.y = y
        self.d = d
        self.w = w
        self.x = x
        self.m = x.shape[0]

    def poly_feat(self, x):
        px = []
        for i in range(2, self.d+1):
            px.append(x[:,1:]**i)
        return np.squeeze(np.array(px)).T

    def add_poly_feat(self, x):
        if self.d < 2:
            pass
        elif self.d == 2:
            zz = np.reshape(self.poly_feat(x), (self.poly_feat(x).shape[0], 1))
            x = np.append(x, zz, axis=1)
        else:
            x = np.append(x, self.poly_feat(x), axis=1)
        return x

    def initiate_weights(self,  process="zero"):
        if process == "zero":
            self.w = np.zeros((self.x.shape[1], 1))
        elif process == "random":
            self.w = np.random.randn(self.x.shape[1], 1)

    def cost_fn(self, reg):
        J = (np.sum((self.y - self.x@self.w)**2) + reg*np.sum(self.w[1:]@self.w[1:].T))/(2*self.m)
        return J

    def gradients(self, x, y, w, reg):
        dw = np.zeros(w.shape)
        m = x.shape[0]
        dw[0] = x[:,0].T@(x@w - y)/m
        dw[1:] = (x[:,1:].T@(x@w - y) + reg*w[1:])/m
        return dw

    def train(self, alpha=0.001, batch_size=1, n=1000, reg=0):
        cost = np.zeros((n,2))
        self.x = self.add_poly_feat(self.x)
        self.initiate_weights()
        ind = np.random.permutation(self.m)
        X = self.x[ind]
        Y = self.y[ind]
        for i in range(n):
            for j in range(self.m//batch_size):
                x = X[j*batch_size:(j+1)*batch_size, :]
                y = Y[j*batch_size:(j+1)*batch_size, :]
                self.w = self.w-alpha*self.gradients(x, y, self.w, reg)
            if self.m % batch_size != 0:
                x = X[self.m//batch_size*batch_size:self.m]
                y = Y[self.m//batch_size*batch_size:self.m]
            self.w = self.w - alpha*self.gradients(x, y, self.w, reg)
            print("Cost after", i+1, "Iterations : ", self.cost_fn(reg))
            cost[i,0] = i+1
            cost[i,1] = self.cost_fn(reg)
        print("\nFinal weights after training : ", self.w)
        return cost

    def predict(self, x):
        x = self.add_poly_feat(x)
        y_ = x@self.w
        return y_
