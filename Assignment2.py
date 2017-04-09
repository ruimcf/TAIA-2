# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:49:22 2017

@author: Pepe
"""
import copy

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
N = 16
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
    
    '''given data points, one can compute the covariance matrix (needs to transpose to get observations through rows and points through columns: '''
    
    CovarianceMatrix = np.cov(data.T) #or just do np.cov(data, rowvar=False)
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor()
    gpr.fit(data[:,0:2], data[:,2])
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)

    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict([[B1[i,j], B2[i,j]]])    
    return Z

V(data)

# Create plot
fig = plt.figure(figsize=(10,6))
fig.suptitle('Gaussian Process Regression', fontsize=20)

ax = axes3d.Axes3D(fig)
x1 = np.linspace(0,1.01,100)
x2 = np.linspace(0,1.01,100)
B1, B2 = np.meshgrid(x1, x2, indexing='xy')
ax.plot_surface(B1, B2, V(data), rstride=10, cstride=5, alpha=0.4)
ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r')

ax.set_xlabel('x')
ax.set_xlim(0,1)
ax.set_ylabel('y')
ax.set_ylim(ymin=0)
ax.set_zlabel('z');

plt.show()

'''h() computes the attractiveness of the central point of a given zone of the map w.r.t. the starting point'''
''' it outputs the central point and its value '''
def h(zone,currentpoint,data,speed):
    #print('freeZone',zone)
    #x = np.zeros(1,2)
    x = [0., 0.]
    '''create point in center of zone using [(x3,y3)-(x1,y1)]/2 + (x1,y1)'''
    x[0] = (zone[2][0] - zone[0][0]) / 2 + zone[0][0]#point between bottomRight and bottomLeft
    x[1] = (zone[2][1] - zone[0][1]) / 2 + zone[0][1]#point between upLeft and bottomRight
    '''evaluate point '''
    Z_pred = V(data)[99*int(x[0])][99*int(x[1])] #get predicted value of x from gaussian distribution
    
    #print('start',currentpoint)
    dist = math.sqrt((x[0]-currentpoint[0])**2 + (x[1]-currentpoint[1])**2)#ditância entre dado ponto e ponto actual onde se encontra o navio
    t_viagem = dist/speed
    #print('t_viagem',t_viagem)
    #print('Z_pred',Z_pred)
    lucro = max(Z_pred - t_viagem/100, 0 ) #max between 0 and profit, to avoid negative value
    #print('lucro',lucro)
    return [x, lucro]


'''h([3,2],[0,0],1)'''

'''Prepare expediction'''

def splitZones(zonesList):
    newZonesList = []
    for zone in zonesList:
        DownLeft = zone[0]
        DownRight = zone[1]
        UpRight = zone[2]
        UpLeft = zone[3]
        
        midDown = [(DownRight[0] - DownLeft[0])/2 + DownLeft[0], (DownRight[1] - DownLeft[1])/2 + DownLeft[1]]
        midRight = [(UpRight[0] - DownRight[0])/2 + DownRight[0], (UpRight[1] - DownRight[1])/2 + DownRight[1]]
        midUp = [(UpLeft[0] - UpRight[0])/2 + UpRight[0], (UpLeft[1] - UpRight[1])/2 + UpRight[1]]
        midLeft = [(DownLeft[0] - UpLeft[0])/2 + UpLeft[0], (DownLeft[1] - UpLeft[1])/2 + UpLeft[1]]
        center = [(UpRight[0] - DownLeft[0])/2 + DownLeft[0], (UpRight[0] - DownLeft[0])/2 + DownLeft[0]]
        '''
        midDown = [(downRight[0] - downLeft[0])/2, downRight[1]]
        midRight = [upRight[0], (upRight[1] - downRight[1])/2]
        midUp = [(upRight[0] - upLeft[0])/2, upRight[1]]
        midLeft = [upLeft[0], (upLeft[1] - downLeft[1])/2]
        center = [midDown[0], midLeft[1]]
        '''
        newZonesList.append([DownLeft, midDown, center, midLeft])
        newZonesList.append([midDown, DownRight, midRight, center])
        newZonesList.append([center, midRight, UpRight, midUp])
        newZonesList.append([midLeft, center, midUp, UpLeft])

    return newZonesList

def removeZones2k(freeZonesList, positionsList):
    for zone in freeZonesList:
        for pos in positionsList:#check if pos inside zone, if yes, remove from list
            print(pos)
    return freeZonesList

'''StopDiving should say if the division in zones made so far already accounts for freeZones or not'''
def StopDividing(data,zones):
    point_in_zone = []
    for zone in zones:
        for point in data:
            '''if point inside zone:'''
            if zone[0][0] < point[0] and zone[1][0] > point[0] and zone[0][1] < point[1] and zone[2][1] > point[1]:
                point_in_zone.append(point)
                
        if point_in_zone == []: #if number of points in zone is zero, after all points of data have been tested, then return True
            #print('zone',zone)
            #print('stop dividing')
            return True #this means it is OK to stop dividing
    #print('continue dividing')
    return False #in the end of testing for all zones, since there is none with zero points, then return False (continue dividing)

'''FreeZones(data,zones) returns a list of zones that are free given data and zones'''
def FreeZones(data,zones):
    freeZones = []
    zones_with_points = []
    nrZones = len(zones)
    zoneSide = 1./math.sqrt(nrZones)
    '''assign for each point a zone without iterating through each zone: '''
    for point in data:
        x_projection_proportion = point[0]/zoneSide #valor que ajudará a encontrar a posição horizontal da zona do ponto
        y_projection_proportion = point[1]/zoneSide #valor que ajudará a encontrar a posição vertical da zona do ponto
        #print('number of zones',nrZones)
        for i in range(int(math.sqrt(nrZones))):
            if i < x_projection_proportion and i+1 > x_projection_proportion:
                for j in range(int(math.sqrt(nrZones))):
                    if j < y_projection_proportion and j+1 > y_projection_proportion:
                        '''obter cada ponto da zona encontrada usando a proporcao das projeccoes do ponto'''
                        p1 = [i*zoneSide, j*zoneSide]
                        p2 = [(i+1)*zoneSide, j*zoneSide]
                        p3 = [(i+1)*zoneSide, (j+1)*zoneSide]
                        p4 = [i*zoneSide, (j+1)*zoneSide]
                        correspondingZone = [p1,p2,p3,p4]
        zones_with_points.append(correspondingZone)
        
    for i in zones:
        if i not in zones_with_points:
            #print('free zone in function',i)
            freeZones.append(i)
    '''if all zones have points, return []'''
    if len(zones_with_points) == len(zones):
        return [[], zones_with_points]
    
    return [freeZones, zones_with_points]
              
    


zone = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]] #initial zone (all the grid)
zones=[zone]
'''avoid setting freeZones = zones, because alterations to freeZones might cause alterations to zones'''
freeZones = [[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]] #initially no zone has been explored


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
            #print('StopDividing',StopDividing(data,zones))
            #while StopDividing(data,zones) is False:
            #print('Zonas pre split', zones)
            zones = splitZones(zones)
            #print('Zonas pos split',zones)
            #freeZones = copy.deepcopy(zones) #deepcopy used to avoid changes of list 'zones' when 'freeZones' is changed
            freeZones = FreeZones(data,zones)[0]
            #print('freeZones', freeZones)
            #print('zones with points', FreeZones(data,zones)[1])
        '''visitar todas as freezones '''
        for freeZone in freeZones:#define number of points to search beforehand
            #print(len(freeZones))
            #print('chegou aqui')
            #if profit < h(freeZone,start,data,speed)[1]:#user np.any if ncessary
            profit = h(freeZone,start,data,speed)[1]
            profits.append(profit) #save each expected profit
            #print('profit',profit)
            chosenZone = freeZone #this means this zone is chosen
            freeZones.remove(chosenZone) #remove chosenZone
            #print('freeZones', freeZones)
            #print('chosenZone',chosenZone)
            chosenPoint = h(freeZone,start,data,speed)[0]
            if chosenPoint == positions[-1]:
                break
            positions.append(chosenPoint) #save chosen points for probing
                #print('chosenPoint',chosenPoint)
                #print('data',data)
        #print('freeZones',freeZones)
        '''freeZones = removeZones2k(free)'''
        # data = np.append(data,[[start[0],start[1],V(data)[start[0]][start[1]]]],axis=0)
        
        #print('positions',positions[len(data):])
        
        i+=1
    #print('Positions',positions)
    #print('Profits',profits)
    
'''plot visited points'''
positions_x = []
positions_y = []
for i in range(len(positions)):
    positions_x.append(positions[i][0])
    positions_y.append(positions[i][1])

plt.plot(positions_x, positions_y, 'ro')
plt.axis([0, 1, 0, 1])
plt.show()

def FinalEstimation(F, data):
    data = data_with_true_values
    for i in range(len(data)):
        data[i,2] = F(data[i,0],data[i,1])
        
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor()
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
