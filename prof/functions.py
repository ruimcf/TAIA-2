# PROBLEM: matplotlib does not seem to display two surfaces correctly...

from __future__ import division
from __future__ import print_function
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class _data:
    def __init__(self):
        self.x,self.y,self.sigma,self.a = {},{},{},{}
        self.x[1],self.y[1] = 0.1,0.8
        self.a[1] = 100.
        self.sigma[1] = 0.1

        self.x[2],self.y[2] = 0.75,0.75
        self.a[2] = 75.
        self.sigma[2] = 0.05

        self.x[3],self.y[3] = 0.5,0.5
        self.a[3] = 50
        self.sigma[3] = 0.1

        self.x[4],self.y[4] = 0.8,0.05
        self.a[4] = 75
        self.sigma[4] = 0.05

        self.x[5],self.y[5] = 0.3,0.1
        self.a[5] = 25
        self.sigma[5] = 0.15
data = _data()

def _f(x,y,n):
    """parameters:
        - x: coordinate where to evaluate the function
        - y:
        - n: number of "centers" from 'data' to use
        """
    value = 0.
    for i in range(1,n+1):
        value += data.a[i] * math.exp( - ((x-data.x[i])/data.sigma[i])**2/2 - ((y-data.y[i])/data.sigma[i])**2/2)
    return value

def f1(x,y): return _f(x,y,1)
def f2(x,y): return _f(x,y,2)
def f3(x,y): return _f(x,y,3)
def f4(x,y): return _f(x,y,4)
def f5(x,y): return _f(x,y,5)


def plot(f,n,filename=None):
    # prepare data
    X = np.arange(0, 1.01, 1./n)
    Y = np.arange(0, 1.01, 1./n)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros((n+1, n+1))
    W = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            Z[i, j] = f(X[i, j], Y[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z(x,y)')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)

    plt.show()


if __name__ == "__main__":
    fs = [f1, f2, f3, f4, f5]
    for i in range(5):
        f = fs[i]
        plot(f,100)
