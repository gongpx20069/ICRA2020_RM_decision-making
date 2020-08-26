
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import datetime
import math
from math import cos, sin
from map import Map
import random

# from pynput import keyboard
# from pynput.keyboard import Key
"""
本环境为1v2的环境
"""

SCREEN_WIDTH = 808
SCREEN_HEIGHT = 448
FRAME_PER_SECOND = 20
BR1_POS = (50,50)
BR2_POS = (50,400)
RR1_POS = (760, 300)
RR2_POS = (760, 50)


class RMEnv(gym.Env):
    """
    Dscription:

    Obervation:


    Actions:
    Type: Discrete(6)
    Num   Action
    0     shot
    1     up
    2     down
    3     left
    4     right
    5    not move
    """
    metadata = {'render.models': ['human'],
                'video.frames_per_scond': 20}
    # 3600 帧

    def __init__(self):
        self.bluerobot1 = Robot(*BR1_POS)
        self.redrobot1 = Robot(*RR1_POS, color=(1, 0, 0), team='red')
        self.redrobot2 = Robot(*RR2_POS, rotation=math.pi*3/2,color=(1, 0, 0), team='red')
        self.action_space = spaces.Tuple((spaces.Discrete(6),spaces.Discrete(6),spaces.Discrete(6)))
        self.seed()
        self.viewer = None
        self.state = None
        map = Map()
        self.map = map.generate_map()
        self.costmap = map.generate_cost_map()
        self.buff_list = self.init_buff()
        self.wall_list = self.init_wall()

    def init_buff(self):
        # value 2:红方回血区， 3:红方弹丸补给区 4:蓝方回血区 5:蓝方弹丸补给区  6:禁止射击区 7:禁止移动区
        buff_info = [(46, 275, 1),
                     (186, 161, 2),
                     (808 - 404, 448 - 40, 3),
                     (808 - 46, 448 - 275, 4),
                     (808 - 186, 448 - 161, 5),
                     (404, 40, 6),
                     ]
        buff_list = []
        for info in buff_info:
            buff_list.append(Buff(*info))
        return buff_list

    def init_wall(self):
        wall_width = 100
        wall_height = 20
        wall_info = [(50, 338, wall_width, wall_height),
                     (190, 224, wall_width - 20, wall_height),
                     (110, 50, wall_height, wall_width),
                     (404, 103.5, wall_width, wall_height),
                     (404, 224, 25, 25, math.pi / 4),
                     (404, 448 - 103.5, wall_width, wall_height),
                     (808 - 110, 448 - 50, wall_height, wall_width),
                     (808 - 190, 224, wall_width - 20, wall_height),
                     (808 - 50, 110, wall_width, wall_height)
                     ]
        wall_list = []
        for info in wall_info:
            wall_list.append(Wall(*info))
        return wall_list

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  # TODO:
        return [seed]

    def get_state(self):
        return (*self.bluerobot1.get_state(), *self.redrobot1.get_state(), *self.redrobot2.get_state(),*self.order_list)

    def step(self, action):
        self.start += 1
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # bluerobot1
        r1 = self.bluerobot1.move(action[0], map=self.map, costmap=self.costmap, start=self.start, buff_list=self.buff_list,
                             robot_list=[self.redrobot1,self.redrobot2], viewer=self.viewer)



        r3 = self.redrobot1.move(action[1], map=self.map, costmap=self.costmap, start=self.start, buff_list=self.buff_list,
                                  robot_list=[self.bluerobot1,self.redrobot2], viewer=self.viewer)

        r4 = self.redrobot2.move(action[2], map=self.map, costmap=self.costmap, start=self.start, buff_list=self.buff_list,
                                  robot_list=[self.bluerobot1,self.redrobot1], viewer=self.viewer)

        self.state = self.get_state()

        # if self.start % 100 ==0:
        #     if self.bluerobot1.hp >(self.redrobot1.hp + self.redrobot2.hp):
        #         r1 += 100
        #         r3 -= 100
        #         r4 -= 100
        #     elif self.bluerobot1.hp <(self.redrobot1.hp + self.redrobot2.hp):
        #         r1 -= 100
        #         r3 += 100
        #         r4 += 100
        #     else:
        #         r1 -= 10
        #         r3 -= 10
        #         r4 -= 10

        if self.start % (FRAME_PER_SECOND*60) == 0:
            self.random_buff()


        if self.start>=FRAME_PER_SECOND*60*3 or self.bluerobot1.hp<=0 or (self.redrobot1.hp<=0 and self.redrobot2.hp<=0 ):

            done = True
        else:
            done = False



        reward = (r1, r3, r4)

        return np.array(self.state), reward, done, {}

    def distance(self, x1,y1):
        x2 = 404
        y2 = 224
        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        d = 1- d/391
        r = d*10-5
        return math.floor(r)

    def reset(self):
        if self.viewer is not None:
            temp = []
            for geom in self.viewer.geoms:
                if geom.remove == False:
                    temp.append(geom)

            self.viewer.geoms = temp
        self.bluerobot1.reset(*BR1_POS, self.viewer)
        self.redrobot1.reset(*RR1_POS, self.viewer)
        self.redrobot2.reset(*RR2_POS, self.viewer)



        self.random_buff()
        self.start= 0
        self.state = self.get_state()

        return np.array(self.state)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 808
        screen_height = 448
        if self.viewer is None:
            self.bluerobot1.is_render = True
            self.redrobot1.is_render = True
            self.redrobot2.is_render = True

            self.viewer = rendering.Viewer(screen_width, screen_height)

            if self.state is None:
                return None
            self.bluerobot1.render_robot(rendering, self.viewer)
            self.redrobot1.render_robot(rendering, self.viewer)
            self.redrobot2.render_robot(rendering, self.viewer)
            self.render_wall(rendering, self.viewer)
            self.render_buff(rendering, self.viewer)

        self.render_hp_bullet(self.bluerobot1.hp, self.bluerobot1.bullet_num,self.redrobot1.hp, self.redrobot2.bullet_num,rendering,self.viewer)
        # self.render_red_hp_bullet(self.enemyrobot.hp, rendering, self.viewer)
        self.bluerobot1.render_update_robot(rendering, self.viewer)
        self.redrobot1.render_update_robot(rendering, self.viewer)
        self.redrobot2.render_update_robot(rendering, self.viewer)



        if self.start % (FRAME_PER_SECOND*60) == 0:
            self.render_update_buff(rendering, self.viewer)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_update_buff(self, rendering, viewer):
        temp = []
        for geom in viewer.geoms:
            if not hasattr(geom,'buff'):
                temp.append(geom)
        viewer.geoms = temp

        for buff in self.buff_list:
            buff.render_buff(rendering, viewer)


    def render_hp_bullet(self,hp1, bullet1,hp2, bullet2,rendering, viewer):
        temp = []
        for geom in viewer.geoms:
            if not hasattr(geom, 'is_hp_bullet'):
                temp.append(geom)
        viewer.geoms = temp

        hp1 /= 10
        hp1 = int(hp1)
        hp1 = max(0,hp1)
        l, r, t, b = -10 / 2, 10 / 2, hp1,0
        hp_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        hp_render.remove = True
        hp_render.is_hp_bullet = True
        trans = rendering.Transform((5,130))
        hp_render.add_attr(trans)
        viewer.add_geom(hp_render)
        hp_render.set_color(0,0,1)


        l, r, t, b = -10 / 2, 10 / 2, bullet1, 0
        bullet_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        bullet_render.remove = True
        bullet_render.is_hp_bullet = True
        trans = rendering.Transform((15, 130))
        bullet_render.add_attr(trans)
        viewer.add_geom(bullet_render)
        bullet_render.set_color(0, 0, 1)


        hp2 /= 10
        hp2 = int(hp2)
        hp2 = max(hp2, 0)
        l, r, t, b = -10 / 2, 10 / 2, hp2, 0
        hp_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        hp_render.remove = True
        hp_render.is_hp_bullet = True
        trans = rendering.Transform((803, 130))
        hp_render.add_attr(trans)
        viewer.add_geom(hp_render)
        hp_render.set_color(1, 0, 0)

        l, r, t, b = -10 / 2, 10 / 2, bullet2, 0
        bullet_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        bullet_render.remove = True
        bullet_render.is_hp_bullet = True
        trans = rendering.Transform((793, 130))
        bullet_render.add_attr(trans)
        viewer.add_geom(bullet_render)
        bullet_render.set_color(1, 0, 0)


    def render_wall(self, rendering, viewer):
        for wall in self.wall_list:
            wall.render_wall(rendering, viewer)

    def render_buff(self, rendering, viewer):
        for buff in self.buff_list:
            buff.render_buff(rendering, viewer)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def random_buff(self):
        # value 2:红方回血区， 3:红方弹丸补给区 4:蓝方回血区 5:蓝方弹丸补给区  6:禁止射击区 7:禁止移动区
        order = [0, 1, 2]
        random.shuffle(order)
        self.order_list = []
        self.buff_list[order[0]].value = 2
        self.buff_list[order[0]].color = (1, 0, 0)


        self.buff_list[order[0] + 3].value = 4
        self.buff_list[order[0] + 3].color = (0, 0, 1)

        self.buff_list[order[1]].value = 3
        self.buff_list[order[1]].color = (1, 1, 0)

        self.buff_list[order[1] + 3].value = 5
        self.buff_list[order[1] + 3].color = (0, 1, 1)

        self.buff_list[order[2]].value = 6
        self.buff_list[order[2]].color = (0, 1, 0)

        self.buff_list[order[2] + 3].value = 7
        self.buff_list[order[2] + 3].color = (0, 0, 0)

        self.order_list.append(self.buff_list[order[0]].x)
        self.order_list.append(self.buff_list[order[0]].y)
        self.order_list.append(0 if self.buff_list[order[0]].activate else 1)

        self.order_list.append(self.buff_list[order[1]].x)
        self.order_list.append(self.buff_list[order[1]].y)
        self.order_list.append(0 if self.buff_list[order[1]].activate else 1)

        self.order_list.append(self.buff_list[order[0]+3].x)
        self.order_list.append(self.buff_list[order[0]+3].y)
        self.order_list.append(0 if self.buff_list[order[0]+3].activate else 1)

        self.order_list.append(self.buff_list[order[1]+3].x)
        self.order_list.append(self.buff_list[order[1]+3].y)
        self.order_list.append(0 if self.buff_list[order[1]+3].activate else 1)

        self.order_list.append(self.buff_list[order[2]].x)
        self.order_list.append(self.buff_list[order[2]].y)
        self.order_list.append(0 if self.buff_list[order[2]].activate else 1)

        self.order_list.append(self.buff_list[order[2]+3].x)
        self.order_list.append(self.buff_list[order[2]+3].y)
        self.order_list.append(0 if self.buff_list[order[2]+3].activate else 1)

        for buff in self.buff_list:
            buff.activate = False
            self.costmap = buff.update_map(self.costmap)



class Bullet():
    def __init__(self, x, y, speed=5, rotation=0.0, color=(0, 0, 0)):
        self.x = x
        self.y = y
        self.speed = speed
        self.rotation = rotation
        self.color = color
        self.bullet = None

    def render_bullet(self, rendering, viewer):
        self.bullet = rendering.make_circle(2)
        self.bullet.remove = True
        self.trans = rendering.Transform()
        self.bullet.add_attr(self.trans)
        viewer.add_geom(self.bullet)
        self.bullet.set_color(*self.color)

    def update_bullet(self):
        self.x += self.speed * cos(self.rotation)
        self.y += self.speed * sin(self.rotation)
        # self.x = min(self.x, SCREEN_WIDTH-1)
        # self.y = min(self.y, SCREEN_HEIGHT-1)

    def render_update_bullet(self):
        self.trans.set_translation(self.x, self.y)
        self.trans.set_rotation(self.rotation)



    def _is_shot_the_robot(self, self_robot, robot_list):
        for robot in robot_list:
            d = math.sqrt((self.x - robot.x) ** 2 + (self.y - robot.y) ** 2)
            if d < 20:  # 子弹打到了机器人身上
                robot.hp -= 40
                if self_robot.team == robot.team:
                    return (True, -10)
                else:
                    return (True, 10)
        return (False, 0)

    def _is_valid(self, map):
        if self.x < 0 or self.x >= SCREEN_WIDTH or self.y < 0 or self.y >= SCREEN_HEIGHT:  # 超出了地图范围
            return False
        elif map[int(self.x), int(self.y)] == 1:  # 碰到了墙
            return False
        else:
            return True


class Robot():
    def __init__(self, x, y, speed=5, rspeed=0.1, width=50, height=50, rotation=0.0, color=(0, 0, 1), team='blue'):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rotation = rotation
        self.team = team
        self.speed = speed  # forward or backward
        self.rspeed = rspeed  # rotation speed
        self.hp = 2000
        self.bullet_num = 50
        self.color = color

        self.pole_width = 5
        self.pole_height = 40
        self.bullet_list = []
        self.is_render = False  # 是否开启界面显示

        self.no_shot = False
        self.no_move = False

    def reset(self, x, y, viewer):
        self.x = x
        self.y = y
        self.hp = 2000
        self.bullet_num = 50
        self.rotation = 0.0
        self.no_shot = False
        self.no_move = False
        for bullet in self.bullet_list:
            if self.is_render and bullet.bullet in viewer.geoms:
                viewer.geoms.remove(bullet.bullet)
            self.bullet_list.remove(bullet)
        self.bullet_list = []

    def get_state(self):
        return (self.x, self.y, self.rotation, self.hp, self.bullet_num)

    def move(self, action, map, costmap, start, buff_list, robot_list, viewer=None):
        r = 0
        if action == 1 and not self.no_move:  # forward
            x = self.x + self.speed * cos(self.rotation)
            y = self.y + self.speed * sin(self.rotation)
            if self._is_invalid(int(x), int(y), costmap) or self._is_collision(x,y,robot_list):  # 碰到墙
                # r = -5
                pass
            else:
                self.x = x
                self.y = y
                _, r = self._is_into_buff(int(x), int(y), costmap, start, buff_list, robot_list)

        elif action == 2 and not self.no_move:  # backward
            x = self.x - self.speed * cos(self.rotation)
            y = self.y - self.speed * sin(self.rotation)
            if self._is_invalid(int(x), int(y), costmap) or self._is_collision(x, y,robot_list): # 碰到墙
                # r = -5
                pass
            else:
                self.x = x
                self.y = y
                _, r = self._is_into_buff(int(x), int(y), costmap, start, buff_list, robot_list)

        elif action == 3:  # left rotation
            self.rotation += self.rspeed
            self.rotation %= (math.pi*2)
            r = 0
        elif action == 4:  # right rotation
            self.rotation -= self.rspeed
            self.rotation %= (math.pi*2)
            r = 0
        elif action == 0 and self.bullet_num > 0 and not self.no_shot:
            bullet = Bullet(self.x, self.y, rotation=self.rotation, color=(0, 0, 0))
            self.bullet_list.append(bullet)
            self.bullet_num -= 1
            r = -1

        if self.no_move and start-self.no_move_time >= 10*FRAME_PER_SECOND:
            self.no_move = False

        if self.no_shot and start - self.no_shot_time >=10*FRAME_PER_SECOND:
            self.no_shot = False

        for bullet in self.bullet_list:
            bullet.update_bullet()

            shot, reward = bullet._is_shot_the_robot(self, robot_list)
            r += reward

            if not bullet._is_valid(map) or shot:
                if self.is_render and bullet.bullet in viewer.geoms:
                    viewer.geoms.remove(bullet.bullet)
                self.bullet_list.remove(bullet)

        return r

    def _is_collision(self,x,y, robot_list):
        for robot in robot_list:
            d = math.sqrt((x - robot.x) ** 2 + (y - robot.y) ** 2)
            if d < 40: #发生碰撞
                return True
        return False

    def _is_invalid(self, x, y, costmap):
        if costmap[x, y] == 1:  # 碰到墙了
            return True
        else:
            return False

    def _is_into_buff(self, x, y, costmap, start, buff_list, robot_list):
        # robot_list.append(self)
        # value 2:红方回血区， 3:红方弹丸补给区 4:蓝方回血区 5:蓝方弹丸补给区  6:禁止射击区 7:禁止移动区
        buff_info = [2, 3, 4, 5, 6, 7]
        if costmap[x, y] not in buff_info:
            return False, 0

        for buff in buff_list:
            if buff.value == costmap[x, y]:
                b = buff
        if b.activate:
            return (True, 0)

        if costmap[x, y] == 2:
            for robot in robot_list:
                if robot.team == 'red':
                    robot.hp += 200
            if self.team == 'red':
                self.hp += 200
                r = 10
            else:
                r = -10
        elif costmap[x, y] == 3:
            for robot in robot_list:
                if robot.team == 'red':
                    robot.bullet_num += 100
            if self.team == 'red':
                self.bullet_num += 100
                r = 10
            else:
                r = -10
        elif costmap[x, y] == 4:
            for robot in robot_list:
                if robot.team == 'blue':
                    robot.hp += 200
            if self.team == 'blue':
                self.hp += 200
                r = 10
            else:
                r = -10
        elif costmap[x, y] == 5:
            for robot in robot_list:
                if robot.team == 'blue':
                    robot.bullet_num += 100
            if self.team == 'blue':
                self.bullet_num += 100
                r = 10
            else:
                r = -10
        elif costmap[x, y] == 6:
            self.no_shot = True
            self.no_shot_time = start
            r = -10

        elif costmap[x, y] == 7:
            self.no_move = True
            self.no_move_time = start
            r = -10

        b.activate = True
        return (True, r)

    def render_robot(self, rendering, viewer):
        l, r, t, b = -self.height / 2, self.height / 2, self.width / 2, -self.width / 2
        robot = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        robot.remove = False
        self.trans = rendering.Transform()
        robot.add_attr(self.trans)

        viewer.add_geom(robot)
        robot.set_color(*self.color)
        l, r, t, b = -self.pole_height / 2, self.pole_height / 2, self.pole_width / 2, -self.pole_width / 2
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.remove = False
        pole.set_color(*self.color)

        viewer.add_geom(pole)

        self.pole_trans = rendering.Transform()
        pole.add_attr(self.pole_trans)

    def render_update_robot(self, rendering, viewer):
        self.trans.set_translation(self.x, self.y)
        self.trans.set_rotation(self.rotation)

        pole_x = self.x + (self.pole_height / 2) * cos(self.rotation)
        pole_y = self.y + (self.pole_height / 2) * sin(self.rotation)

        self.pole_trans.set_translation(pole_x, pole_y)
        self.pole_trans.set_rotation(self.rotation)

        for bullet in self.bullet_list:
            if bullet.bullet not in viewer.geoms:
                bullet.render_bullet(rendering, viewer)
            else:
                bullet.render_update_bullet()


class Buff():
    def __init__(self, x, y, no, value=2, width=46, height=40, color=(255, 179, 0)):
        # value 2:红方回血区， 3:红方弹丸补给区 4:蓝方回血区 5:蓝方弹丸补给区  6:禁止射击区 7:禁止移动区
        self.x = x
        self.y = y
        self.no = no
        self.width = width
        self.height = height
        self.color = color
        self.value = value
        self.activate = False  # 默认不被激活，激活之后无法再次被激活

    def render_buff(self, rendering, viewer):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        self.buff = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        self.buff.remove = False
        self.buff.buff = True
        self.trans = rendering.Transform()
        self.buff.add_attr(self.trans)
        self.buff.set_color(*self.color)
        viewer.add_geom(self.buff)

        self.trans.set_translation(self.x, self.y)

    def update_map(self, costmap):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        l, r, t, b = int(l), int(r), int(t), int(b)
        costmap[self.x + l:self.x + r, self.y + b:self.y + t] = self.value
        return costmap


class Wall():
    def __init__(self, x, y, width, height, rotation=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rotation = rotation
        self.color = (0, 0, 0)

    def render_wall(self, rendering, viewer):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        self.wall = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        self.wall.remove = False
        self.trans = rendering.Transform()
        self.wall.add_attr(self.trans)

        viewer.add_geom(self.wall)
        self.wall.set_color(*self.color)

        self.trans.set_translation(self.x, self.y)
        self.trans.set_rotation(self.rotation)


# def on_press(key):
#     print(key)
#     print(type(key))
#     try:
#         print('alphanumeric key  {0} pressed'.format(key.char))
#     except AttributeError:
#         print('special key {0} pressed'.format(key))
#
#
# def on_release(key):
#     print('{0} released'.format(key))
#     if key == keyboard.Key.esc:
#         return False


if __name__ == '__main__':
    # env = gym.make("CartPole-v0")
    # env.reset()
    # ACTION_KEYS = [Key.left, Key.right]
    #
    # def on_press(key):
    #     render = True
    #     if key in ACTION_KEYS:
    #         s_, r, d, _ = env.step(ACTION_KEYS.index(key))
    #         if render: env.render()
    #         if d: env.reset()
    # with keyboard.Listener(on_press=on_press) as listener:
    #     listener.join()

    env = RMEnv()
    ob = env.reset()
    done = False
    # ACTION_KEYS = [Key.space,Key.up, Key.down, Key.left, Key.right]
    # def on_press(key):
    #     if key in ACTION_KEYS:
    #         action = env.action_space.sample()[1]
    #         ob, reard, done, _ = env.step((ACTION_KEYS.index(key),action))
    #         print(ob)
    #         env.render()
    #         if done:
    #             env.reset()
    #
    # with keyboard.Listener(on_press=on_press) as listener:
    #     listener.join()


    while not done:
        action = env.action_space.sample()
        print("action:", action)
        ob, reard, done, _ = env.step(action)
        print(ob)
        print(type(ob))
        # time.sleep(10)
        if done:
            break
        env.render()
        # time.sleep(1)
    env.close()
