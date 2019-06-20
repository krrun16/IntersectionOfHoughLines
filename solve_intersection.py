import cv2
import numpy as np
import matplotlib.pyplot as plt

class LineSegment(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class LinearEquation2D(object):
    def __init__(self, xc, yc, b):
        self.xc = xc
        self.yc = yc
        self.b = b

    def plot(self, miny=-100, maxy=100, npoints=100, draw='-r'):
        x = np.linspace(minx, maxx, npoints)
        y = (self.xc * x - self.b) / (-self.yc)
        
        plt.plot(x, y, draw)


def lineFromSegment(seg):
    m = (seg.y2 - seg.y1)/(seg.x2 - seg.x1)
    xc = m
    yc = -1
    b = m * seg.x1 - seg.y1
    return LinearEquation2D(xc, yc, b)

def leq(xc, yc, b):
    return LinearEquation2D(xc, yc, b)

def solve2D(eqns):
    A = np.array([[e.xc, e.yc] for e in eqns], dtype='float')
    B = np.array([e.b for e in eqns], dtype='float')
    return cv2.solve(A, B, flags=cv2.DECOMP_SVD)


#camera = (0, -100)

# example 1: in lane
segments = [LineSegment(-100, -100, -20, 20), # one on left
         LineSegment(100, -100, 20, 20), # one on right
         LineSegment(-200, -100, -40, 20), # another on left
         LineSegment(-50, -50, 35, -35)] # outlier (outnumbered by inliers
lines = [lineFromSegment(s) for s in segments]
_, intersect = solve2D(lines)
print (intersect)

[l.plot(draw=d) for l,d in zip(lines, 3*['-r']+1*['-g'])]
plt.plot([intersect[0]], [intersect[1]], color='black', marker='+', markersize=12)
plt.grid()
plt.show()

# example 2: veering right
segments = [LineSegment(-100, -100, -120, 20), # one on left
         LineSegment(100, -100, -20, 20), # one on right
         LineSegment(-150, -00, -170, 20), # another on left
         LineSegment(-75, -75, 50, -50)] # outlier (outnumbered by inliers
lines = [lineFromSegment(s) for s in segments]
_, intersect = solve2D(lines)
print (intersect)

[l.plot(draw=d) for l,d in zip(lines, 3*['-r']+1*['-g'])]
plt.plot([intersect[0]], [intersect[1]], color='black', marker='+', markersize=12)
plt.grid()
plt.show()


# example 3: in lane lots of outliers (shadows)
segments = [LineSegment(-100, -100, -20, 20), # one on left
         LineSegment(100, -100, 20, 20), # one on right
         LineSegment(-200, -100, -40, 20), # another on left
         LineSegment(-75, -75, 75, -75), # outliers shadows
         LineSegment(-100, -100, 100, -100),
         LineSegment(-25, -25, 25, -25),
         LineSegment(-50, -50, 50, -50)] 
lines = [lineFromSegment(s) for s in segments]
_, intersect = solve2D(lines)
print (intersect)

[l.plot(draw=d) for l,d in zip(lines, 3*['-r']+4*['-g'])]
plt.plot([intersect[0]], [intersect[1]], color='black', marker='+', markersize=12)
plt.grid()
plt.show()

# example 4: veering right with lots of outliers (shadows)
for ybase in [-100, -75, -50, -25, 0]:
    segments = [LineSegment(-100, -100, -120, 20), # one on left
             LineSegment(100, -100, -20, 20), # one on right
             LineSegment(-150, -100, -170, 20), # another on left
             LineSegment(-75, 75+ybase, 75, 75+ybase), # outliers shadows
             LineSegment(-100, 100+ybase, 100, 100+ybase),
             LineSegment(-25, 25+ybase, 25, 25+ybase),
             LineSegment(-50, 50+ybase, 50, 50+ybase)] 
    lines = [lineFromSegment(s) for s in segments]
    _, intersect = solve2D(lines)
    print (intersect)

    [l.plot(draw=d) for l,d in zip(lines, 3*['-r']+4*['-g'])]
    plt.plot([intersect[0]], [intersect[1]], color='black', marker='+', markersize=12)
    plt.grid()
    plt.show()

"""
# y = -x + 1
# y = x + 1
A = np.array([[-1,-1],[1,-1]], dtype='float')
B = np.array([-1,-1], dtype='float')
print ( cv2.solve(A,B) )

# three lines with middle intersection
A = np.array([[-10,-1],[1,-2],[2,-1]], dtype='float')
B = np.array([2,-5,-7], dtype='float')
print ( cv2.solve(A,B, flags=cv2.DECOMP_SVD) )
"""
