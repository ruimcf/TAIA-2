# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from random import random
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
    return np.sin(10*(x+y))

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
    #CovarianceMatrix = np.cov(data.T) #or just do np.cov(data, rowvar=False)
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
    #kernel = Exponentiation(kernel, exponent) #use in combination with any of previous kernels
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(data[:,0:2], data[:,2])
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)
    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict(np.asarray([[B1[i,j], B2[i,j]]]).reshape(1, -1))
    return gpr

'''h() computes the attractiveness of the central point of a given zone of the map w.r.t. the starting point'''
''' it outputs the central point and its value '''
def h(zone, data, model):
    Z_pred = model.predict(np.asarray(zone.center).reshape(1, -1)) #get predicted value of x from gaussian distribution
    #points that define the zone:
    Z_pred2 = model.predict(np.asarray(zone.downLeft).reshape(1, -1))
    Z_pred3 = model.predict(np.asarray(zone.downRight).reshape(1, -1))
    Z_pred4 = model.predict(np.asarray(zone.upRight).reshape(1, -1))
    Z_pred5 = model.predict(np.asarray(zone.upLeft).reshape(1, -1))
    '''compute covariance matrix of 5 points within this region: the points that define the region, and the central point'''
    dataPoints = []
    for point in zone.getPoints():
        dataPoints.append(point+model.predict(np.asarray(point).reshape(1, -1)))
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
            return True #this means it is OK to stop dividing
    return False #in the end of testing for all zones, since there is none with zero points, then return False (continue dividing)

def FreeZones(data,zones):
    freeZones = []
    zones_with_points = []
    nrZones = len(zones)
    zoneSide = 1./math.sqrt(nrZones)
    for point in data:
        x_projection_proportion = point[0]/zoneSide #valor que ajudará a encontrar a posição horizontal da zona do ponto
        y_projection_proportion = point[1]/zoneSide #valor que ajudará a encontrar a posição vertical da zona do ponto
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
            freeZones.append(zone)
        else:
            zones_with_points.append(zone)

    return [freeZones, zones_with_points]


def planner(X, z):
    N=len(X)
    data = np.zeros((N,3))
    dataPoints = []
    for i in range(N):
        x, y = X[i]
        data[i,0] = x
        data[i,1] = y
        data[i,2] = z[i]
        dataPoints.append([x, y])

    model = V(data)
    zones = [Zone([0.,0.],[1.,0.],[1.,1.],[0.,1.])]
    freeZones = FreeZonesQuadratic(data,zones)[0]
    #Split the Grid until we have free zones
    while freeZones == []:
        zones = splitZones(zones)
        freeZones = FreeZonesQuadratic(data,zones)[0]
    while True:
        attractiveness = []
        bestZones = []
        newPositions = []
        print("Nova iteracao")
        for zone in freeZones:
            attractiveness.append(h(zone, data, model))
        for i in range(0, len(attractiveness)):
            newBestZone = freeZones.pop(attractiveness.index(max(attractiveness)))
            bestZones.append(newBestZone)
            attractiveness.remove(max(attractiveness))
            newPositions.append(newBestZone.center)

            route, cost = tsp(newPositions)
            print("Tempo com {} pontos: {}".format(len(newPositions), cost))
            if cost > inst.T:
                #Ultrapassamos o tempo maximo mas podemos tentar com outro valor, talvez mais perto
                newPositions.remove(newBestZone.center)
                bestZones.remove(newBestZone)

        if FreeZonesQuadratic(dataPoints+newPositions, zones)[0] == []:
            zones = splitZones(zones)
            freeZones = FreeZonesQuadratic(data, zones)[0]
        else:
            # Atingimos a solução maxima
            route, cost = tsp(newPositions)
            return route

# basic function that returns the time to check some points
def tsp(points):
    route = []
    cost = 0
    currentPoint = [0, 0]
    pointsList = copy.deepcopy(points)
    for i in range(0,len(points)):
        min = 10
        for point in pointsList:
            distance = dist(currentPoint[0], currentPoint[1], point[0], point[1])
            if distance != 0 and distance < min:
                min = distance
                nextPoint = point
        route.append(nextPoint)
        cost += dist(currentPoint[0], currentPoint[1], nextPoint[0], nextPoint[1])
        pointsList.remove(nextPoint)
        currentPoint = nextPoint
    route.append([0, 0])
    cost += dist(currentPoint[0], currentPoint[1], 0, 0)
    cost += len(points)
    return route, cost
#------------------------------------------------------
#Actualizar o Kernel

def Error(z, model, x, y):
    return abs(z - model.predict(np.asarray([x,y]).reshape(1, -1)))

def Vtest(data,kernel):
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(data[:,0:2], data[:,2])
    return gpr

def KernelUpdate(data, kernel):
    part1 = data[:int(len(data)*0.5),:]
    part2 = data[int(len(data)*0.5):,:]
    E = []
    kernels = []
    avgErrors = []
    kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) #grade: 7
    kernel2 = C(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0)) #grade: 5
    kernel3 = 1**2*RBF(length_scale = 1) #grade: 1
    kernel4 = 0.594**2*RBF(length_scale = 0.279) #grade: 1
    kernel5 = 1**2*Matern(length_scale=1,nu=1.5) #grade: 1
    kernel6 = 0.609**2*Matern(length_scale=0.484, nu = 1.5) #grade: 7
    kernel7 = 1**2*RationalQuadratic(alpha=0.1,length_scale=1) #grade: 6
    kernel8 = 0.594**2*RationalQuadratic(alpha=1e+05, length_scale=0.279) #grade 1
    kernel9 = 1**2*ExpSineSquared(length_scale=1,periodicity=3) #grade: 1
    kernel10 = 0.799**2*ExpSineSquared(length_scale=0.791,periodicity=2.87) #grade: 1
    kernel11 = 0.799**2*ExpSineSquared(length_scale=0.791,periodicity=2.87) #grade: ?? estalactites e estalgmites por todo o lado
    kernel12 = 0.316**2*DotProduct(sigma_0=1)**2 #grade = 4 (pringle shape)
    kernel13 = 0.316**2*DotProduct(sigma_0=0.368)**2 #grade:2 (too simple)
    exponent = 2
    kernel14 = Exponentiation(kernel, exponent) #use in combination with any of previous kernels
    kernels.append(kernel1)
    kernels.append(kernel2)
    kernels.append(kernel3)
    kernels.append(kernel4)
    kernels.append(kernel5)
    kernels.append(kernel6)
    kernels.append(kernel7)
    kernels.append(kernel8)
    kernels.append(kernel9)
    kernels.append(kernel10)
    kernels.append(kernel11)
    kernels.append(kernel12)
    kernels.append(kernel13)
    kernels.append(kernel14)
    for kernel in kernels:
        testModel = Vtest(part1, kernel)
        for point in part2:
            E.append(Error(point[2], testModel, point[0], point[1]))
        avgErrors.append(np.mean(E))
    return kernels[avgErrors.index(min(avgErrors))]

def estimator(X, z, mesh):
    N=len(X)
    data = np.zeros((N,3))
    for i in range(N):
        x, y = X[i]
        data[i,0] = x
        data[i,1] = y
        data[i,2] = z[i]
    newKernel = KernelUpdate(data, C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))
    gpr = GaussianProcessRegressor(kernel = newKernel)
    gpr.fit(X, z)
    z = []
    for (x_,y_) in mesh:
        GP = gpr.predict(np.asarray([(x_,y_)]).reshape(1, -1))
        z.append(GP)
    return z

def FinalEstimation(F, data, kernel):
    Data = np.zeros((len(data),3))
    for i in range(len(Data)):
        Data[i,0] = data[i][0]
        Data[i,1] = data[i][1]
        Data[i,2] = F(data[i][0],data[i][1])
        #now data has true values
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(Data[:,0:2], Data[:,2])
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)
    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict(np.asarray([[B1[i,j], B2[i,j]]]).reshape(1, -1))
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Final Gaussian Process Regression with kernel: ' + str(kernel), fontsize=20)
    ax = axes3d.Axes3D(fig)
    ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
    ax.scatter3D(Data[:,0], Data[:,1], Data[:,2], c='r')
    ax.set_xlabel('x')
    ax.set_xlim(0,1)
    ax.set_ylabel('y')
    ax.set_ylim(ymin=0)
    ax.set_zlabel('z');
    plt.show()
    return Z

def UpdateData(data, route):
    newData = np.copy(data)
    for point in route:
        np.append(newData, [[point[0], point[1], g(point[0], point[1])]], axis=0)
    return newData

def plotSplitZones():
    zones = [Zone([0.,0.],[1.,0.],[1.,1.],[0.,1.])] #initial zone (all the grid)
    PositionsX = []
    PositionsY = []
    for i in range(3):
        zones = splitZones(zones)
        for zone in zones:
            for point in zone.getPoints():
                PositionsX.append(point[0])
                PositionsY.append(point[1])
        plt.plot(PositionsX, PositionsY, 'ro')
        plt.axis([0, 1, 0, 1])
        plt.show()
    import matplotlib.pyplot
    free1 = []
    free2 = []
    zones = [Zone([0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5])]
    freeZones = [Zone([0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5])]
    zones = splitZones(zones)
    while FreeZonesQuadratic(data,zones)[0] == []:
        zones = splitZones(zones)
        freeZones = FreeZonesQuadratic(data,zones)[0]
    for freezone in freeZones:
        for point in freezone.getPoints():
            free1.append(point[0])
            free2.append(point[1])
    x_axis = np.append(data[:,0], free1)
    y_axis = np.append(data[:,1], free2)
    plt.plot(data[:,0], data[:,1],'ro')
    plt.plot(free1, free2)
    plt.axis([0, 1, 0, 1])
    plt.show()

def plotKernels():
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Gaussian Process Regression  with kernel: ' + str(kernel), fontsize=20)
    plt.hold(True)
     #grade: 7
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
    #exponent = 2
    #kernel = Exponentiation(kernel, exponent) #use in combination with any of previous kernels
    gpr = GaussianProcessRegressor(kernel = kernel)
    gpr.fit(data[:,0:2], data[:,2])
    ax = axes3d.Axes3D(fig)
    x1 = np.linspace(0,1.01,100)
    x2 = np.linspace(0,1.01,100)
    B1, B2 = np.meshgrid(x1, x2, indexing='xy')
    Z = np.zeros((x2.size, x1.size))
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = gpr.predict(np.asarray([[B1[i,j], B2[i,j]]]).reshape(1, -1))
    ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c='r')
    ax.set_xlabel('x')
    ax.set_xlim(0,1)
    ax.set_ylabel('y')
    ax.set_ylim(ymin=0)
    ax.set_zlabel('z');

def plotVisitedPoints(route):
    newPositions_x = []
    newPositions_y = []
    for point in route:
        newPositions_x.append(point[0])
        newPositions_y.append(point[1])
    plt.plot(newPositions_x, newPositions_y, 'ro')
    plt.axis([0, 1, 0, 1])
    plt.title("Visited Points")
    plt.show()

def createMesh():
    mesh = []
    for i in range(101):
        for j in range(101):
            x, y = i/100., j/100.
            mesh.append((x,y))
    return mesh

if __name__ == "__main__":
    data = init()
    route = planner(data[:,0:2], data[:,2])
    print("Route: ",route)
    newData = UpdateData(data, route)
    mesh = createMesh()
    z = estimator(data[:,0:2], data[:,2], mesh)
    value = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        value += abs(g(x,y) - float(z[i]))
    print(value)

    plotKernels()
    iinalGaussian = FinalEstimation(g, newData, KernelUpdate(newData, C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))))
    plotVisitedPoints(route)


