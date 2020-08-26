import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Map():
    def __init__(self):
        self.width = 808
        self.height = 448
        self.robot_size = 40

    def generate_map(self):
        self.map = np.zeros((self.width,self.height),dtype=np.int8)
        wall_width = 100
        wall_height = 20
        wall_info = [(50, 338, wall_width, wall_height),
                     (190, 224, wall_width - self.robot_size/2, wall_height),
                     (110, 50, wall_height, wall_width),
                     (404, 103, wall_width, wall_height),  # 103.5换成了103
                     (404, 224, 25, 25),
                     (404, 448 - 103, wall_width, wall_height),  # 103.5换成了103
                     (808 - 110, 448 - 50, wall_height, wall_width),
                     (808 - 190, 224, wall_width - self.robot_size/2, wall_height),
                     (808 - 50, 110, wall_width, wall_height)
                     ]
        self.wall_list = []
        for info in wall_info:
            self.wall_list.append(Wall(*info))

        for i,wall in enumerate(self.wall_list):
            wall.show_wall(self.map)

        # buff_info = [(46, 275),
        #              (186, 161),
        #              (404, 40),
        #              (808 - 46, 448 - 275),
        #              (808 - 186, 448 - 161),
        #              (808 - 404, 448 - 40),
        #              ]
        # buff_list = []
        # for info in buff_info:
        #     buff_list.append(Buff(*info))
        #
        # for buff in buff_list:
        #     buff.show_buff(self.map)

        return self.map

    def generate_cost_map(self):
        self.costmap = self.generate_map()
        for wall in self.wall_list:
            l, r, t, b = -(wall.width+self.robot_size) / 2, (wall.width+self.robot_size) / 2, (wall.height+self.robot_size) / 2, -(wall.height+self.robot_size) / 2
            l, r, t, b = int(l), int(r), int(t), int(b)
            self.costmap[max(wall.x + l,0):wall.x + r, max(wall.y + b,0):wall.y + t] = 1
        mask = np.zeros((self.width, self.height),dtype=np.int8)
        l,r,b,t = self.robot_size/2,(self.width-self.robot_size/2),self.robot_size/2,(self.height-self.robot_size/2)
        l, r, b, t = int(l), int(r), int(b), int(t)
        mask[l:r, b:t] = 1
        self.costmap[mask==0] = 1
        return self.costmap


class Buff():
    def __init__(self,x , y, width = 46, height=40):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def show_buff(self, costmap):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        l, r, t, b = int(l),int(r),int(t),int(b)
        costmap[self.x+l:self.x+r,self.y+b:self.y+t] = 200
        return costmap


class Wall():
    def __init__(self,x , y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def show_wall(self, map):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        l, r, t, b = int(l),int(r),int(t),int(b)
        map[self.x+l:self.x+r,self.y+b:self.y+t] = 1



if __name__ == '__main__':
    map = Map()
    a = map.generate_cost_map()
    print(a)
    im = Image.fromarray(a)
    im.show()
