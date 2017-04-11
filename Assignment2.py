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


#plot function to predict
fig = plt.figure()
ax = fig.gca(projection='3d')
#plt.hold(True)
x_surf=np.arange(0, 1, 0.01)    # generate a mesh
y_surf=np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = g(x_surf,y_surf)
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot);    # plot a 3d surface plot

plt.show()

N = 16 #number of data points to start with
#X = np.zeros((N,2))
#y = np.zeros((N))
data = np.zeros((N,3))#in the form array[[x1,y1,z1],...,[xn,yn,zn]]
for i in range(N):
    x1=random()
    x2=random()
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = abs(g(data[i,0],data[i,1]))
N=len(data)

#plot data points
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
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, Exponentiation, ConstantKernel as C
    
    '''given data points, one can compute the covariance matrix (needs to transpose to get observations through rows and points through columns: '''
    
    CovarianceMatrix = np.cov(data.T) #or just do np.cov(data, rowvar=False)
    '''kernels functions examples'''
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) #grade: 7
    #kernel = C(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0)) #grade: 5
    #kernel = 1**2*RBF(length_scale = 1) #grade: 1
    #kernel = 0.594**2*RBF(length_scale = 0.279) #grade: 1
    #kernel = 1**2*Matern(length_scale=1,nu=1.5) #grade: 1
    #kernel = 0.609**2*Matern(length_scale=0.484, nu = 1.5) #grade: 7
    #kernel = 1**2*RationalQuadratic(alpha=0.1,length_scale=1) #grade: 6
    #kernel = 0.594**2*RationalQuadratic(alpha=1e+05, length_scale=0.279) #grade 1
    #kernel = 1**2*ExpSineSquared(length_scale=1,periodicity=3) #grade: 1
    #kernel = 0.799**2*ExpSineSquared(length_scale=0.791,periodicity=2.87) #grade: 1
    #kernel = 0.799**2*ExpSineSquared(length_scale=0.791,periodicity=2.87) #grade: ?? estalactites e estalgmites por todo o lado
    #kernel = 0.316**2*DotProduct(sigma_0=1)**2 #grade = 4 (pringle shape)
    #kernel = 0.316**2*DotProduct(sigma_0=0.368)**2 #grade:2 (too simple)
    exponent = 2
    kernel = Exponentiation(kernel, exponent) #use in combination with any of previous kernels
    
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(data[:,0:2], data[:,2])
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)

    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict([[B1[i,j], B2[i,j]]])    
    return Z

V(data)

def PCA(data):
    '''given data points, one can compute the covariance matrix (needs to transpose to get observations through rows and points through columns: '''
    CovarianceMatrix = np.cov(data.T) #or just do np.cov(data, rowvar=False)
    # eigenvectors and eigenvalues from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(CovarianceMatrix)
    
    for i in range(len(eig_val_cov)):
        eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
        #assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
    
        print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
        print(40 * '-')

  # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    print('eig_pairs',eig_pairs)
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print('eig_pairs',eig_pairs)
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i[0])
        
    #Choosing 2 eigenvectors with the largest eigenvalues
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
    print('Matrix W', matrix_w)

    #transform the data points into the new space with the matrix transformation (transformed = data*W)
    #NOTE: data is Nx3-dimensional and W is 3x2 dimensional, transformed is Nx2 dimensional
    transformed = np.dot(data, matrix_w)
    
    CovarianceMatrixTransformed =np.cov(transformed.T)
    
    #transformed = transformed*(-1)
    print('transformed', transformed)
    print('dim transformed', len(transformed))
    #plot results:
    plt.plot(transformed[:,0],transformed[:,1],'o',markersize=7,color='blue',alpha=0.5, label='Transformed points')
    #plt.plot(data[:,0],data[:,1],'^',markersize=7,color='red',alpha=0.5, label='data') #NOTE: it makes no sense to compare in the same 2x2 plot, since the subspace is different
    #plt.plot(transformed[0,0:int(len(data)/2)], transformed[1,0:int(len(data)/2)], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    #plt.plot(transformed[0,int(len(data)/2):len(data)], transformed[1,int(len(data)/2):len(data)], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')
    
    plt.show()
    
    # ALTERNATIVE WAY TO COMPUTE THE SAME THING:
    #with linear PCA
    '''
    from sklearn.decomposition import KernelPCA, PCA as sklearnPCA  
    
    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(data)
    
    plt.plot(sklearn_transf[0:int(len(data)/2),0],sklearn_transf[0:int(len(data)/2),1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(sklearn_transf[int(len(data)/2):int(len(data)),0], sklearn_transf[int(len(data)/2):int(len(data)),1], '^', markersize=7, color='red', alpha=0.5, label='class2')
    
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
    
    plt.show()
    '''
    return [transformed, matrix_w, CovarianceMatrix, eig_val_cov, eig_vec_cov]

#PCA(data)

'''h() computes the attractiveness of the central point of a given zone of the map w.r.t. the starting point'''
''' it outputs the central point and its value '''
def h(zone,currentpoint,data,speed):
    #print('freeZone',zone)
    #x = np.zeros(1,2)
    x = [0., 0.]
    '''create point in center of zone using [(x3,y3)-(x1,y1)]/2 + (x1,y1)'''
    '''NOTE: this is where we choose low variance region points'''
    x[0] = (zone[2][0] - zone[0][0]) / 2 + zone[0][0]#point between bottomRight and bottomLeft
    x[1] = (zone[2][1] - zone[0][1]) / 2 + zone[0][1]#point between upLeft and bottomRight
    '''evaluate points '''
    Z_pred = V(data)[99*int(x[0])][99*int(x[1])] #get predicted value of x from gaussian distribution
    #points that define the zone:
    Z_pred2 = V(data)[99*int(zone[0][0])][99*int(zone[0][1])]
    Z_pred3 = V(data)[99*int(zone[1][0])][99*int(zone[1][1])]
    Z_pred4 = V(data)[99*int(zone[2][0])][99*int(zone[2][1])]
    Z_pred5 = V(data)[99*int(zone[3][0])][99*int(zone[3][1])]
    
    '''compute covariance matrix of 5 points within this region: the points that define the region, and the central point'''
    dataPoints = []
    for point in zone:
        dataPoints.append(point)
    dataPoints.append([x[0], x[1], Z_pred])#prepare dataPoints for PCA 
    CovarianceMatrix = np.cov(map(list, zip(*dataPoints))) #transposed dataPoints
    eigenValues, eigenVectors = np.linalg.eig(CovarianceMatrix) #get eigenvalues of covariance matrix
    #if the eigenvalues are big, then there is strong relation between variables, which means high covariance.
    #we are aiming to zones with low variance, and therefore low eigenvalues.
    
    #print('start',currentpoint)
    dist = math.sqrt((x[0]-currentpoint[0])**2 + (x[1]-currentpoint[1])**2)#ditância entre dado ponto e ponto actual onde se encontra o navio
    t_viagem = dist/speed
    #print('t_viagem',t_viagem)
    #print('Z_pred',Z_pred)
    lucro = max(Z_pred - t_viagem/100, 0 ) #max between 0 and profit, to avoid negative value
    repulsiveness = eigenValues[-1]*t_viagem/np.mean([Z_pred, Z_pred2, Z_pred3, Z_pred4, Z_pred5]) 
    attractiveness = eigenValues[-1]/(t_viagem*np.mean([Z_pred, Z_pred2, Z_pred3, Z_pred4, Z_pred5]))
    #EXPLICACAO DA REPULSIVENESS: o valor proprio associado a Z é tanto maior quanto maior for a variancia de Z.
    #Escolhidos 5 pontos numa zona, é possivel entao determinar a variancia de Z com um certo erro.
    # por isso, quanto maior o valor proprio, pior.
    # quanto maior o tempo de viagem, pior.
    # quanto maior o valor estimado dos pontos na zona, melhor (daí a divisão).
    
    #print('lucro',lucro)
    return [x, lucro, attractiveness, repulsiveness]


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
        center = [(UpRight[0] - DownLeft[0])/2 + DownLeft[0], (UpRight[1] - DownLeft[1])/2 + DownLeft[1]]
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

#plot splitZones:
zone = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]] #initial zone (all the grid)
zones=[zone]
PositionsX = []
PositionsY = []
for i in range(3):
    zones = splitZones(zones)
    for zone in zones:
        for point in zone:
            PositionsX.append(point[0])
            PositionsY.append(point[1])
    plt.plot(PositionsX, PositionsY, 'ro')
    plt.axis([0, 1, 0, 1])
    plt.show()

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
        #print('point', )
        #print('corresponding zone',correspondingZone)
        zones_with_points.append(correspondingZone)
        
    for i in zones:
        if i not in zones_with_points:
            #print('free zone in function',i)
            freeZones.append(i)
    '''if all zones have points, return []'''
    if len(zones_with_points) == len(zones):
        return [[], zones_with_points]
    
    return [freeZones, zones_with_points]
              
def FreeZonesQuadratic(data, zones):
    freezones = []
    zones_with_points = []
    


zone = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]] #initial zone (all the grid)
zones=[zone]
'''avoid setting freeZones = zones, because alterations to freeZones might cause alterations to zones'''
freeZones = [[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]] #initially no zone has been explored


if __name__ == "__main__":
    

    positions = data[:,0:2].tolist()
    newPositions = []
    profits = []
    start = [0,0]
    speed = 1
    t=2
    T=5
    i=0
    
    while i < T:
        profit = 0
        values_of_h = []
        #print('freeZones',freeZones)
        if freeZones == []:
            #print('entrou aqui')
            #print('StopDividing',StopDividing(data,zones))
            #while StopDividing(data,zones) is False:
            #print('Zonas pre split', zones)
            zones = splitZones(zones)
            #print('Zonas pos split',zones)
            
            while FreeZones(data,zones)[0] == []:
                zones = splitZones(zones)
            freeZones = FreeZones(data,zones)[0]
            #freeZones = copy.deepcopy(zones) #deepcopy used to avoid changes of list 'zones' when 'freeZones' is changed    
            #print('new freeZones', freeZones)
            #print('freeZones', freeZones)
            #print('zones with points', FreeZones(data,zones)[1])
        #visitar todas as freezones 
        for freeZone in freeZones:#define number of points to search beforehand
            attract = h(freeZone,start,data,speed)[2]#repulsiveness of this particular zone
            values_of_h.append(attract)
            #print('repulse', repulse)
        #print('values of h', values_of_h)
        chosenZone = freeZones[values_of_h.index(max(values_of_h))]#freeZone corresponding to value in values_of_h
        chosenPoint = h(chosenZone,start,data,speed)[0]
        #print('chosen point',chosenPoint)
        if chosenPoint == positions[-1]: #if there are repetitions
            break
        positions.append(chosenPoint) #save chosen points for probing
        newPositions.append(chosenPoint)
        profits.append(max(values_of_h))
        freeZones.remove(chosenZone)
        start = chosenPoint
            #print(len(freeZones))
            #print('chegou aqui')
            #if profit < h(freeZone,start,data,speed)[1]:#user np.any if ncessary
            #profit = h(freeZone,start,data,speed)[1]
            #profits.append(profit) #save each expected profit
                         #print('profit',profit)
            #chosenZone = freeZone #this means this zone is chosen
            #freeZones.remove(chosenZone) #remove chosenZone
            #print('freeZones', freeZones)
            #print('chosenZone',chosenZone)
                         #chosenPoint = h(freeZone,start,data,speed)[0]
                #print('chosenPoint',chosenPoint)
                #print('data',data)
        #print('freeZones',freeZones)
        #freeZones = removeZones2k(free)
        data = np.append(data,[[chosenPoint[0],chosenPoint[1],V(data)[chosenPoint[0]][chosenPoint[1]]]],axis=0)
        
        #print('positions',positions[len(data):])
        
        i+=1

#print('Positions',positions)
print('new positions', newPositions)
#print('Profits',profits)
#print('positions length', len(positions))
#print('profits length', len(profits))

# Create plot of Gaussian Regression
fig = plt.figure(figsize=(10,6))
fig.suptitle('Gaussian Process Regression', fontsize=20)
plt.hold(True)
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

#plt.show()

#plot visited points
positions_x = []
positions_y = []
for i in range(len(positions)):
    positions_x.append(positions[i][0])
    positions_y.append(positions[i][1])
newPositions_x = []
newPositions_y = []
for i in range(len(newPositions)):
    newPositions_x.append(newPositions[i][0])
    newPositions_y.append(newPositions[i][1])
    
plt.plot(positions_x, positions_y, 'ro')
plt.plot(newPositions_x, newPositions_y, 'ro', c=[1,1,1])
#ax.scatter3D(newPositions_x, newPositions_y, np.zeros((1,len(newPositions_x))),c='r')
#plt.plot(newPositions_x, newPositions_y,'ro')
plt.axis([0, 1, 0, 1])
plt.show()

def FinalEstimation(F, data):
    for i in range(len(data)):
        data[i,2] = F(data[i,0],data[i,1])
        #now data has true values
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
