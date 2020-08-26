from PRM_AStar import PRM
from PRM_AStar import Point
import RoboMaster
from map import Map
import random

# prm = PRM(k=6,rand=0.9995)
prm = PRM(k=6,rand=0.999)
# prm.creatPRM()
import math

class Utils():
    def __init__(self):
        map = Map()
        self.map = map.generate_map().astype('float')
        self.action = {"forward": 1,
                  "backward": 2,
                  "left": 3,
                  "right": 4,
                  "stop": 5}
    # def check_middle_obstacle(self, x1, y1, x2, y2):
    #     '''
    #     检测两点中间是否有障碍物，若有则True，若无则False
    #     '''
    #     return self.map[int((x1 + x2) / 2)][int((y1 + y2) / 2)] == 1
    def check_obstacle(self, x1, y1, x2, y2):
        '''
        检测两点之间是否有障碍物，如果则True，若无则False
        '''
        # return True
        mid_x,mid_y = int((x1+x2)/2),int((y1+y2)/2)
        if self.map[mid_x][mid_y] >0.8:
            return True
        elif mid_x == x1 or mid_x == x2 or mid_y == y1 or mid_y== y2:
            return False
        else:
            return self.check_obstacle(x1,y1,mid_x,mid_y) or self.check_obstacle(mid_x,mid_y,x2,y2)

    def get_escape_point(self, enemy_x, enemy_y):
            if enemy_x > 404 and enemy_y > 224:
                # 敌人位置第一象限
                return 186, 80
            elif enemy_y > 224:
                # 敌人位置第二象限
                return 569, 103
            elif enemy_x > 404:
                # 敌人位置第四象限
                return 212, 326
            else:
                # 敌人位置第三象限
                return 569, 328


    def get_action(self, x, y, r, x1, y1, rt, mt):
        """
        :param x: robot.x
        :param y: robot.y
        :param r: robot.rotation
        :param x1: taget.x
        :param y1: target.y
        :param rt: rotation threshold
        :param mt: move threshold
        :return: action: 1,2,3,4
        """
        # print("self: x,y",x,y,"   target:",x1,y1)

        if x1 == x:
            if y1 > y:
                theta = math.pi/2
            else:  #TODO 有问题
                theta = math.pi *3/2
        elif x1>x:  # 1、4象限
            theta = math.atan((y1 - y) / (x1 - x)) % (math.pi * 2)
            # if y1>=y:
            #     theta = math.atan((y1 - y) / (x1 - x))
            # else:
            #     theta = math.atan((y1 - y) / (x1 - x)) % (math.pi * 2)
        elif x1<x:
            if y1 == y:
                theta = math.pi
            elif y1 > y:
                theta = theta = math.atan((y1 - y) / (x1 - x))%math.pi
            elif y1 < y:
                theta = theta = math.atan((y1 - y) / (x1 - x))+ math.pi
        # theta = math.atan((y1 - y) / (x1 - x))%(math.pi*2)  # 实际的角度
        # print("theta",theta)
        if abs((r - theta)) < rt:  # 前进后退操作
            d = math.sqrt((y1 - y) ** 2 + (x1 - x) ** 2)
            # print("distance:",d)
            # print(theta)
            if d < mt:
                return self.action['stop']
            else:
                return self.action['forward']
                # if r < math.pi:
                #     if y1 > y:
                #         return self.action['forward']
                #     else:
                #         if x1<x:
                #             return self.action['forward']
                #         return self.action['backward']
                # else:
                #     if y1 > y:
                #         return self.action['backward']
                #     else:
                #         return self.action['forward']
        else:  # 旋转操作
            # print(r,theta)
            if theta < math.pi:
                if r > theta  and r< theta+math.pi:
                    return self.action['right']
                else:
                    return self.action['left']
            else:
                if r<theta and r > theta-math.pi:
                    return self.action['left']
                else:
                    return self.action['right']

class Statement(object):
    def __init__(self,ob):
        """
        :param self_x: robot的当前位置x
        :param self_y: robot的当前位置y
        :param object_x: 目标点x
        :param object_y: 目标点y
        :param r:  robot的旋转角度 rotation
        初始状态为chase状态
        """
        self.ob = ob
        self.utils = Utils()
        self.rt = 0.1
        self.mt = 10
        self.s = {"shoot":0,"chase":1,"escape":2, "addblood":3, "addbullet":4}
        self.state = self.s['chase']
        self.obecvt_x = int(ob[5])
        self.obecvt_y = int(ob[6])
        self.ChaseState()
        # prm.drawlines(self_x, self_y, object_x, object_y)

    def run(self, ob, network=None,choose='dfsm'):
        """
        :param self_x: robot的实时位置x
        :param self_y: robot的实时位置y
        :return:
        """
        self.ob = ob
        self_x = ob[0]
        self_y = ob[1]
        self_r = ob[2]

        is_update = False
        # print(self.state,"*"*5)
        # print(len(self.router),"*"*50)
        if len(self.router)>0:
            point = self.router[0]
            action = self.utils.get_action(self_x, self_y, self_r, point.x, point.y, self.rt, self.mt)
            if self.state == self.s['chase'] or self.state == self.s['escape'] or self.state== self.s['addbullet'] or self.state == self.s['addblood']:
                if action == 5:
                    self.router.remove(point)
                    # print("have removed the point")
                    # print(len(self.router))
                    a = random.randint(0,1)
                    if a:
                        self.updata_statement(network,choose)
                        is_update = True
            else: # shoot state
                if action == 5 or action==1 or action ==2:
                    # print("action = 0")
                    action = 0
                    self.updata_statement(network,choose)
                    is_update = True
            return action, is_update
        self.updata_statement(network,choose)
        is_update = True
        action = 5
        return action,is_update


    def updata_statement(self,network,choose = 'dfsm'):
        if choose == 'dfsm':
            ob_n = network.policy_old.nomalization(self.ob)
            self.state = network.policy_old.act(ob_n,self.state,network.memory,self.utils.check_obstacle(self.ob[0],self.ob[1],self.ob[5],self.ob[6]))
            if self.state == 0:
                self.ShootState()
            elif self.state == 1:
                self.ChaseState()
            elif self.state == 2:
                self.EscapeState()
            elif self.state == 3:
                self.AddBloodState()
            elif self.state == 4:
                self.AddBulletState()
        else:
            self_bullet = self.ob[4]
            self_hp = self.ob[3]
            bullet_state = self.ob[21]
            blood_state =self.ob[18]
            self_x = int(self.ob[0])
            self_y = int(self.ob[1])
            enemy_x = int(self.ob[5])
            enemy_y = int(self.ob[6])

            if self_bullet<30 and bullet_state:
                self.state = self.s['addbullet']
                self.AddBulletState()
            else:
                if self_hp<1000 and blood_state:
                    self.state = self.s['addblood']
                    self.AddBloodState()
                else:
                    if self_bullet>10 and self_hp>500:
                        d = math.sqrt((self_x-enemy_x)**2 + (self_y-enemy_y)**2)
                        if not self.utils.check_obstacle(self_x,self_y,enemy_x,enemy_y) and d<200:
                            self.state = self.s['shoot']
                            self.ShootState()
                        else:
                            self.state = self.s['chase']
                            self.ChaseState()

                    else:
                        self.state = self.s['escape']
                        self.EscapeState()


    def ShootState(self):
        enemy_x = int(self.ob[5])
        enemy_y = int(self.ob[6])
        self.router = [Point(enemy_x,enemy_y)]

    def ChaseState(self):
        self_x = int(self.ob[0])
        self_y = int(self.ob[1])
        enemy_x = int(self.ob[5])
        enemy_y = int(self.ob[6])
        # print(self_x,self_y,enemy_x,enemy_y)
        self.router = prm.Run(self_x,self_y,enemy_x,enemy_y)
        self.router.pop()


    def EscapeState(self):
        self_x = int(self.ob[0])
        self_y = int(self.ob[1])
        enemy_x = int(self.ob[5])
        enemy_y = int(self.ob[6])
        object_x, object_y = self.utils.get_escape_point( enemy_x, enemy_y)
        self.router = prm.Run(self_x,self_y,object_x, object_y)
        # prm.drawlines(self_x, self_y,object_x, object_y)

    def AddBloodState(self):
        self_x = int(self.ob[0])
        self_y = int(self.ob[1])
        blood_x = int(self.ob[16])
        blood_y = int(self.ob[17])
        self.router = prm.Run(self_x,self_y,blood_x,blood_y)
        # prm.drawlines(self_x, self_y, blood_x,blood_y)

    def AddBulletState(self):
        self_x = int(self.ob[0])
        self_y = int(self.ob[1])
        bullet_x = int(self.ob[19])
        bullet_y = int(self.ob[20])
        self.router = prm.Run(self_x,self_y,bullet_x,bullet_y)
        # prm.drawlines(self.router)
        # prm.drawlines(self_x, self_y, bullet_x,bullet_y)



if __name__ == '__main__':
    env = RoboMaster.RMEnv()
    ob = env.reset()
    done = False

    st = Statement(int(ob[0]), int(ob[1]), int(ob[5]), int(ob[6]))
    while not done:
        a2 = env.action_space.sample()[1]
        print("x,y,r:",ob[0],ob[1],ob[2])
        a1 = st.run(ob[0],ob[1],ob[2])
        print(a1)
        action = (a1,a2)
        ob, reard, done, _ = env.step(action)
        # print(ob)
        # print(type(ob))

        if done:
            break
        env.render()
        # time.sleep(1)
    env.close()