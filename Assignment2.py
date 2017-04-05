# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:49:22 2017

@author: Pepe
"""
import math
import numpy as np
from sklearn import linear_model
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#import seaborn as sns
# Gaussian Regression
'''generate data'''
def g(x,y):
    return np.sin(10*(x+y))+random()
'''plot function to predict'''
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.hold(True)
x_surf=np.arange(0, 1, 0.01)    # generate a mesh
y_surf=np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = g(x_surf,y_surf)
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot);    # plot a 3d surface plot
N = 100
#X = np.zeros((N,2))
#y = np.zeros((N))
data = np.zeros((N,3))#in the form array[[x1,y1,z1],...,[xn,yn,zn]]
for i in range(N):
    x1=random()
    x2=random()
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = g(data[i,0],data[i,1])
N=len(data)

'''plot data points'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2]);

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_zlabel('z label')

plt.show()
'''function to do gaussian regression given some data'''
def V(data):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor()
    gpr.fit(data[:,0:2], data[:,2])
    #print('data[:,0:1]',data[:,0:2])

    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)

    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))

    for (i,j),v in np.ndenumerate(z):
        Z[i,j] = gpr.predict([[B1[i,j], B2[i,j]]])
    #print('Z',Z)

    # Create plot
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Gaussian Process Regression', fontsize=20)

    ax = axes3d.Axes3D(fig)

    ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r')

    ax.set_xlabel('x')
    ax.set_xlim(0,1)
    ax.set_ylabel('y')
    ax.set_ylim(ymin=0)
    ax.set_zlabel('z');

    plt.show()

    return Z

V(data)
#print('small z',z)
#print('data',data)
#print('predictions',predictions)

#print('Z',Z)
#print('z[2,1]',Z[0,0])

#print('x1',data[:,0])
#print('x2',data[:,1])
#print('y',data[:,2])
#print('data',data)

#Function for attractiveness of points
#Input: qualquer X =[x1,x2] do conjunto de pontos candidatos
#       ponto de posição actual do navio, currentpoint
#       velocidade do navio, speed
#       tempo de sondagem, t, e tempo total de exploração, T

def h(x,currentpoint,data,speed):
    #X=np.zeros((1,2))#transforming point to array
    #X[0,0]=x[0]
    hello = "Ola"
    #X[0,1]=x[1]
    Z_pred = V(data)[x[0]][x[1]] #get predicted value of x from gaussian distribution
    #print('Z_pred',Z_pred)
    print('start',currentpoint)
    dist = math.sqrt((x[0]-currentpoint[0])**2 + (x[1]-currentpoint[1])**2)#ditância entre dado ponto e ponto actual onde se encontra o navio
    t_viagem = dist/speed
    print('t_viagem',t_viagem)
    print('Z_pred',Z_pred)
    lucro = max(Z_pred - t_viagem/100, 0 ) #max between 0 and profit, to avoid negative value
    print('lucro',lucro)
    return lucro


'''h([3,2],[0,0],1)'''

'''Expediction'''

import time

Positions = []
Profits = []
start = [0,0]
speed = 1
t=2
T=10
timeout = time.time()+T
while time.time() < timeout:
    profit = 0
    for i in range(5):#define number of points to search beforehand
        print('candidate' + str(1), i+1)
        candidate = [random(),random()]
        print('candidate',candidate)#substituir por função de v(candidate) (como escolher os candidatos)
        if profit < h(candidate,start,data,speed):#user np.any if ncessary
            profit = h(candidate,start,data,speed)
            start = candidate #this means this candidate is chosen and therefore will be next starting point
            data = np.append(data,[[start[0],start[1],V(data)[start[0]][start[1]]]],axis=0)
            print('data',data)

    Positions.append(candidate) #save chosen points for probing
    Profits.append(profit) #save each expected profit
    time.sleep(t) #time for probing makes the code stop executing for t seconds
print('Positions',Positions)
print('Profits',Profits)
#h([0,0],[0,0],1)
#h([0.8,1],[0,0],1)
#h([0.2,0.4],[0,0],1)

#while T > 100:
    #estudar pontos numa vizinhança do ponto de posição actual com a função h(x,y)
    #escolher o que optimiza a função h
    #acrescentar esse ponto aos dados e computar v(X)








