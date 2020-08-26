from map import Map
import numpy as np
import sys
import copy
import math
from queue import PriorityQueue
import matplotlib.pyplot as plt
import json

class PRM(object):
    def __init__(self,k=4,rand=0.99):
        map = Map()
        map.robot_size = 50
        self.costmap = map.generate_cost_map().astype('float')
        self.mapshape = np.shape(self.costmap)
        self.allpoints = []
        self.start = None
        self.end = None
        self.k = k
        self.rand = rand
        self.astar = AStar()
    def check_middle_obstacle(self,x1,y1,x2,y2):
        '''
        检测两点中间是否有障碍物，若有则True，若无则False
        '''
        mid_x,mid_y = int((x1+x2)/2),int((y1+y2)/2)
        if self.costmap[mid_x][mid_y] >0.8:
            return True
        elif mid_x == x1 or mid_x == x2 or mid_y == y1 or mid_y== y2:
            return False
        else:
            return self.check_middle_obstacle(x1,y1,mid_x,mid_y) or self.check_middle_obstacle(mid_x,mid_y,x2,y2)

    def checkobstacle(self,point):
        list_remove = []
        for adj in point.adjacency:
            if self.check_middle_obstacle(point.x,point.y,adj.x,adj.y):
                list_remove.append(adj)
        for adj in list_remove:
            point.adjacency.remove(adj)
        return point

    def createPRM(self):
        randcostmap = self.costmap>0.8
        randmap = np.random.rand(self.mapshape[0],self.mapshape[1])
        randmap = randmap>self.rand
        randmap[randcostmap] = False
        xs,ys = np.where(randmap==True)
        for i, j in zip(xs,ys):
            self.allpoints.append(Point(i,j))
        for i in range(len(self.allpoints)):
            self.allpoints[i] = self.createadj(self.allpoints[i])
        for i in range(len(self.allpoints)):
            self.allpoints[i] = self.checkobstacle(self.allpoints[i])

    def Run(self,x1, y1, x2, y2):
        self.start = Point(x1,y1)
        self.end = Point(x2,y2)
        start_linear = self.findlinear(self.start)
        end_linear = self.findlinear(self.end)
        router = self.astar.run(start_linear,end_linear,self.allpoints)
        while router == -1:
            self.allpoints.clear()
            self.createPRM()
            start_linear = self.findlinear(self.start)
            end_linear = self.findlinear(self.end)
            router = self.astar.run(start_linear,end_linear,self.allpoints)
        router = router[::-1]
        router.append(self.end)
        return router


    def drawlines(self,router):
        plt.imshow(self.costmap)
        for point in self.allpoints:
            for adj in point.adjacency:
                plt.plot([adj.y,point.y],[adj.x,point.x],color='blue')
        plt.imshow(self.costmap)
        point0 = self.start
        for point in router:
            plt.plot([point0.y,point.y],[point0.x,point.x],color='red')
            point0=point
        plt.show()

    # def planing(self,endP):
    def findlinear(self,point):
        alldis = [self.distance(point,point1) for point1 in self.allpoints]
        sortall = copy.deepcopy(alldis)
        sortall.sort()
        index = alldis.index(min(sortall))
        return self.allpoints[index]

    def createadj(self,point):
        alldis = [self.distance(point,point1) for point1 in self.allpoints]
        sortall = copy.deepcopy(alldis)
        sortall.sort()
        for i in range(self.k+1):
            if sortall[i] !=0:
                index = alldis.index(sortall[i])
                point.adjappend(self.allpoints[index])
        return point

    def distance(self,point1,point2):
        return (point1.x-point2.x)**2+(point1.y-point2.y)**2

    def savegraph(self,filename='prm.json'):
        savelist = []
        for point in self.allpoints:
            savelist.append((float(point.x),float(point.y)))
        with open(filename,'w') as f:
            json.dump(savelist,f)

    def loadgraph(self,filename='prm.json'):
        with open(filename,'r') as f:
            savelist = json.load(f)
        self.allpoints.clear()
        for point in savelist:
            self.allpoints.append(Point(int(point[0]),int(point[1])))
        for i in range(len(self.allpoints)):
            self.allpoints[i] = self.createadj(self.allpoints[i])
        for i in range(len(self.allpoints)):
            self.allpoints[i] = self.checkobstacle(self.allpoints[i])


class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.adjacency = []

    def __eq__(self,other):
        return (self.x+self.y)==(other.x+other.y)
    def __gt__(self,other):
        return (self.x+self.y)>(other.x+other.y)
    def __ge__(self,other):
        return (self.x+self.y)<(other.x+other.y)

    def adjappend(self,P):
        self.adjacency.append(P)


class AStar(object):
    def __init__(self):
        self.pq = PriorityQueue()
    def clearpq(self):
        while not self.pq.empty():
            self.pq.get()

    def distance(self,point1,point2):
        return math.sqrt((point1.x-point2.x)**2+(point1.y-point2.y)**2)

    def run(self,start,end,allpoints):
        router = []
        cost_so_far = {}
        came_from = {}
        self.pq.put(start,0)
        cost_so_far[(start.x,start.y)] = 0
        came_from[(start.x,start.y)] = None
        while not self.pq.empty():
            current = self.pq.get()
            if current == end:
                temp = current
                while temp!= start:
                    router.append(temp)
                    temp = came_from[(temp.x,temp.y)]
                router.append(start)
                self.clearpq()
                return router
            for adj in current.adjacency:
                new_cost = cost_so_far[(current.x,current.y)]+self.distance(current,adj)
                if (adj.x,adj.y) not in cost_so_far or new_cost<cost_so_far[(adj.x,adj.y)]:
                    cost_so_far[(adj.x,adj.y)] = new_cost
                    priority = new_cost+ self.distance(adj,end)
                    self.pq.put(adj,priority)
                    came_from[(adj.x,adj.y)] = current
        return -1



if __name__ == '__main__':
    plan = PRM(k=6,rand=0.999)
    plan.createPRM()
    # plan.savegraph('prm_blue.json')
    plan.drawlines(plan.Run(216,125,647,291))
    # new = PRM(k=6,rand=0.9995)
    # new.loadgraph('prm_blue.json')
    # new.drawlines(new.Run(216,125,647,291))

    # plt.imshow(a)
    # plt.show()