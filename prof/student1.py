import random
import math
from sklearn.gaussian_process import GaussianProcessRegressor

# object for keeping instance data
class INST:
    pass
inst = INST()
inst.t = 1    # time for probing
inst.s = 1    # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0   # depot coordinates
inst.y0 = 0   # depot coordinates

# auxiliary function: euclidean distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# required function: route planning
def planner(X,z):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
    """
    # do something with 'X', 'z'
    route = []   # list of coordinates to visit
    time = 0     # elapsed time
    x0, y0 = (inst.x0,inst.y0)  # initial position (depot)
    xt, yt = (0,0)  # current position
    while True:
        x, y = random.random(), random.random()
        if time + dist(xt,yt,x,y)/inst.s + inst.t + dist(x,y,x0,y0)/inst.s <= inst.T:
            time += dist(xt,yt,x,y)/inst.s + inst.t
            route.append((x,y))
            xt, yt = x, y
        else:
            return route


def estimator(X,z,mesh):
    """estimator: evaluate z at points [(x1,y1), ...., (xK,yK)] based on M=n+N known points:
        - X: list of coordinates [(x1,y1), ...., (xM,yM)]
        - z: list of evaluations of "true" function [z1, ..., zM]
    """
    gpr = GaussianProcessRegressor()
    gpr.fit(X, z)

    z = []
    for (x_,y_) in mesh:
        GP = gpr.predict([(x_,y_)])
        z.append(GP)

    return z