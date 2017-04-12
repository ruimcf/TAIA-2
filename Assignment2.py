# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
from sklearn import linear_model
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, Exponentiation, ConstantKernel as C

# object for keeping instance data
class INST:
    pass
inst = INST()
inst.t = 1    # time for probing
inst.s = 1    # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0   # depot coordinates
inst.y0 = 0   # depot coordinates

class Zone:
    def __init__(self, downLeft, downRight, upRight, upLeft):
        self.downLeft = downLeft
        self.downRight = downRight
        self.upRight = upRight
        self.upLeft = upLeft
        self.midDown = [(downRight[0] - downLeft[0])/2 + downLeft[0], downLeft[1]]
        self.midRight = [downRight[0], (upRight[1] - downRight[1])/2 + downRight[1]]
        self.midUp = [(upLeft[0] - upRight[0])/2 + upRight[0], upRight[1]]
        self.midLeft = [upLeft[0], (downLeft[1] - upLeft[1])/2 + upLeft[1]]
        self.center = [(upRight[0] - downLeft[0])/2 + downLeft[0], (upRight[1] - downLeft[1])/2 + downLeft[1]]
    def split(self):
        newZonesList = []
        newZonesList.append(Zone(self.downLeft, self.midDown, self.center, self.midLeft))
        newZonesList.append(Zone(self.midDown, self.downRight, self.midRight, self.center))
        newZonesList.append(Zone(self.center, self.midRight, self.upRight, self.midUp))
        newZonesList.append(Zone(self.midLeft, self.center, self.midUp, self.upLeft))
        return newZonesList
    def getPoints(self):
        return [self.downLeft, self.downRight, self.upRight, self.upLeft]

# auxiliary function: euclidean distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# example function
def g(x,y):
    return np.sin(10*(x+y))+random()

def init():
    #plot function to predict
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_surf=np.arange(0, 1, 0.01)    # generate a mesh
    y_surf=np.arange(0, 1, 0.01)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = g(x_surf,y_surf)
    ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot);    # plot a 3d surface plot
    plt.show()
    N = 16 #number of data points to start with
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
    return data

def V(data):
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
    return gpr


'''h() computes the attractiveness of the central point of a given zone of the map w.r.t. the starting point'''
''' it outputs the central point and its value '''
def h(zone, data, model):
    Z_pred = model.predict(zone.center) #get predicted value of x from gaussian distribution
    #points that define the zone:
    Z_pred2 = model.predict(zone.downLeft)
    Z_pred3 = model.predict(zone.downRight)
    Z_pred4 = model.predict(zone.upRight)
    Z_pred5 = model.predict(zone.upLeft)
    '''compute covariance matrix of 5 points within this region: the points that define the region, and the central point'''
    dataPoints = []
    for point in zone.getPoints():
        dataPoints.append(point+model.predict(point))
    dataPoints.append(zone.center+Z_pred)#prepare dataPoints for PCA
    CovarianceMatrix = np.cov(np.transpose(np.asarray(dataPoints))) #transposed dataPoints
    eigenValues, eigenVectors = np.linalg.eig(CovarianceMatrix) #get eigenvalues of covariance matrix
    #if the eigenvalues are big, then there is strong relation between variables, which means high covariance.
    #we are aiming to zones with high variance, and therefore high eigenvalues.
    lucro = max(Z_pred, 0 ) #max between 0 and profit, to avoid negative value
    attractiveness = eigenValues[-1]/(np.mean([Z_pred, Z_pred2, Z_pred3, Z_pred4, Z_pred5]))
    #EXPLICACAO DA ATTRACTIVENESS: o valor proprio associado a Z é tanto maior quanto maior for a variancia de Z.
    #Escolhidos 5 pontos numa zona, é possivel entao determinar a variancia de Z com um certo erro.
    # por isso, quanto maior o valor proprio, pior.
    # quanto maior o valor estimado dos pontos na zona, melhor (daí a divisão).
    return attractiveness


def splitZones(zonesList):
    newZonesList = []
    for zone in zonesList:
        zonesToAppend = zone.split()
        for newZone in zonesToAppend:
            newZonesList.append(newZone)
    return newZonesList


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
    freeZones = []
    zones_with_points = []
    for zone in zones:
        points_in_zone = []
        for point in data:
            if zone.downLeft[0] < point[0] and zone.downRight[0] > point[0] and zone.downLeft[1] < point[1] and zone.upRight[1] > point[1]:
                points_in_zone.append(point)
                break #break the data loop, because this zone cannot be a freeZone
        if points_in_zone == []:
            print('zone', zone)
            freeZones.append(zone)
        else:
            zones_with_points.append(zone)

    return [freeZones, zones_with_points]


def planner(X, z):
    N=len(X)
    data = np.zeros((N,3))
    for i in range(N):
        x, y = X[i]
        data[i,0] = x
        data[i,1] = y
        data[i,2] = z[i]

    #model = V(z)
    model = V(data)
    zones = [Zone([0.,0.],[1.,0.],[1.,1.],[0.,1.])]
    #freeZones = FreeZones(zones)[0]
    freeZones = FreeZonesQuadratic(data,zones)[0]
    newPositions = []
    newPositionsValues = []
    profits = []
    bestZones = []
    #Split the Grid until we have free zones
    while freeZones == []:
        zones = splitZones(zones)
        #freeZones = FreeZones(zones)[0]
        freeZones = FreeZonesQuadratic(data,zones)[0]

    while True:
        attractiveness = []
        for zone in freeZones:
            attractiveness.append(h(zone, data, model))
        for i in range(0, len(attractiveness)):
            newBestZone = freeZones.pop(attractiveness.index(max(attractiveness)))
            bestZones.append(newBestZone)
            attractiveness.remove(max(attractiveness))
            newPositions.append(newBestZone.center)
            newPositionsValues.append(V(data).predict([[newBestZone.center[0], newBestZone.center[1]]]))#get value of (x,y) newPoint in list
            data = np.concatenate((data, [[newBestZone.center[0], newBestZone.center[1], newPositionsValues[-1]]]),axis=0)#update data with new point (x,y,z)
            time = tsp(newPositions)
            print("Tempo com {} pontos: {}".format(len(newPositions), time))
            if time > inst.T:
                #Ultrapassamos o tempo maximo mas podemos tentar com outro valor, talvez mais perto
                newPositions.remove(newBestZone.center)
                bestZones.remove(newBestZone)
        
        #if FreeZonesQuadratic(data, zones)[0] == []:
            # Usamos todas as zonas, então devemos continuar a dividir
        #    zones = splitZones(zones)
        #    freeZones = FreeZonesQuadratic(data, zones)[0]
            
        else:
            # Atingimos a solução maxima
            route = []
            for point in newPositions:
                route.append((point[0], point[1]))
            return route;

# basic function that returns the time to check some points
def tsp(points):
    time = 0;
    xt, yt = (0,0)
    for point in points:
        time += dist(xt, yt, point[0], point[1])/inst.s + inst.t
        xt, yt = point[0], point[1]
    time += dist(xt, yt, 0, 0)
    return time;


if __name__ == "__main__":
    data = init()
    route = planner(data[:,0:2], data[:,2])

    #PLOT SPLITZONES:
    zones = [Zone([0.,0.],[1.,0.],[1.,1.],[0.,1.])] #initial zone (all the grid)
    #zones=[zone]
    PositionsX = []
    PositionsY = []
    for i in range(3):
        zones = splitZones(zones)
        print('zones',zones)
        for zone in zones:
            print('zone',zone)
            for point in zone.getPoints():
                PositionsX.append(point[0])
                PositionsY.append(point[1])
        plt.plot(PositionsX, PositionsY, 'ro')
        plt.axis([0, 1, 0, 1])
        plt.show()

    #PLOT FREEZONESQUADRATIC:
    import matplotlib.pyplot
    free1 = []
    free2 = []
    zones = [Zone([0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5])]
    #zones=[zone]
    zones = splitZones(zones)
    print('zones',zones)
    #print('freezones', FreeZonesQuadratic(data,zones)[0])
    while FreeZonesQuadratic(data,zones)[0] == []:
        zones = splitZones(zones)
        freeZones = FreeZonesQuadratic(data,zones)[0]
    for freezone in freeZones:
        print('freezone',freezone)
        for point in freezone:
            free1.append(point[0])
            free2.append(point[1])
    x_axis = np.append(data[:,0], free1)
    y_axis = np.append(data[:,1], free2)
    plt.plot(data[:,0], data[:,1],'ro')
    #matplotlib.pyplot.scatter(free1, free2,color=['green'])
    plt.plot(free1, free2)
    plt.axis([0, 1, 0, 1])
    plt.show()


# Create plot of Gaussian Regression
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Gaussian Process Regression', fontsize=20)
    plt.hold(True)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) #grade: 7
    exponent = 2
    kernel = Exponentiation(kernel, exponent) #use in combination with any of previous kernels
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(data[:,0:2], data[:,2])
    ax = axes3d.Axes3D(fig)
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)
    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict([[B1[i,j], B2[i,j]]])
    ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r')

    ax.set_xlabel('x')
    ax.set_xlim(0,1)
    ax.set_ylabel('y')
    ax.set_ylim(ymin=0)
    ax.set_zlabel('z');


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
