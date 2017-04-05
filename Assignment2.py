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
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(data[:,0:2], data[:,2])
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)

    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict([[B1[i,j], B2[i,j]]])

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

def h(zone,currentpoint,data,speed):
    x = np.zeros((1,2))
    x[0] = (zone[1][0] - zone[0][0]) / 2
    x[1] = (zone[2][1] - zone[1][1]) / 2
    Z_pred = V(data)[x[0]][x[1]] #get predicted value of x from gaussian distribution
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

zone = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
zones=[zone]
freeZones = zones

def splitZones(zonesList):
    newZonesList = []
    for zone in zonesList:
        downLeft = zone[0]
        downRight = zone[1]
        upRight = zone[2]
        upLeft = zone[3]
        midDown = [(downRight[0] - downLeft[0])/2, downRight[1]]
        midRight = [upRight[0], (upRight[1] - downRight[1])/2]
        midUp = [(upRight[0] - upLeft[0])/2, upRight[1]]
        midLeft = [upLeft[0], (upLeft[1] - downLeft[1])/2]
        center = [midDown[0], midLeft[1]]
        newZonesList.append([downLeft, midDown, center, midLeft])
        newZonesList.append([midDown, downRight, midRight, center])
        newZonesList.append([center, midRight, upRight, midUp])
        newZonesList.append([midLeft, center, midUp, upLeft])

    return newZonesList

def removeZones2k(freeZonesList, positionsList):
    for zone in freeZonesList:
        for pos in positionsList:
            #check if pos inside zone, if yes, remove from list
    return freeZonesList

if __name__ == "__main__":
    positions = data[:,0:2].tolist()
    profits = []
    start = [0,0]
    speed = 1
    t=2
    T=10
    i=0
    while i < T:
        profit = 0
        if freeZones == []:
            zones = splitZones(zones)
            freeZones = zones
        for freeZone in freeZones:#define number of points to search beforehand
            if profit < h(freeZone,start,data,speed):#user np.any if ncessary
                profit = h(freeZone,start,data,speed)
                chosen = zone #this means this candidate is chosen and therefore will be next starting point
                print('data',data)

        # data = np.append(data,[[start[0],start[1],V(data)[start[0]][start[1]]]],axis=0)
        positions.append(candidate) #save chosen points for probing
        profits.append(profit) #save each expected profit
        i+=1
    print('Positions',positions)
    print('Profits',profits)


