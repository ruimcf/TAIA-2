import math
import random

# object for keeping instance data
class INST:
    pass
inst = INST()
inst.t = 1    # time for probing
inst.s = 1    # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0   # depot coordinates
inst.y0 = 0   # depot coordinates

EPS = 1.e-6   # for floating point comparisons
INF = float('Inf')

# auxiliary function: euclidean distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def evaluate(f, planner, estimator):
    # test planning part
    X, z = [], []
    for x in [.2, .4, .6, .8]:
        for y in [.2, .4, .6, .8]:
            X.append((x,y))
            z.append(f(x,y))

    route = planner(X,z)
    time = 0     # elapsed time
    (xt,yt) = (inst.x0,inst.y0)
    for (x,y) in route:
        time += dist(xt,yt,x,y)/inst.s + inst.t
        xt, yt = x, y
        X.append((x,y))
        z.append(f(x, y))
        print("probing at ({:8.5g},{:8.5g}) --> \t{:8.5g}".format(x, y, z[-1]))

    time += dist(xt, yt, inst.x0, inst.y0) / inst.s
    if time > inst.T + EPS:
        return INF    # route takes longer than time limit

    # test forecasting part
    mesh = []
    for i in range(101):
        for j in range(101):
            x, y = i/100., j/100.
            mesh.append((x,y))
    z = estimator(X,z,mesh)
    value = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        value += abs(f(x,y) - float(z[i]))

    return value

        
if __name__ == "__main__":
    from functions import f1 as f
    from student1 import planner,estimator
    random.seed(0)
    diff = evaluate(f, planner, estimator)
    print("student's evaluation:\t{:8.7g}".format(diff))

    # # for reading trend from csv file:
    # import csv
    # import gzip
    # with gzip.open("data_2017_newsvendor.csv.gz", 'rt') as f:
    #     reader = csv.reader(f)
    #     data = [int(t) for (t,) in reader]
