import torch
import torch.nn as nn
import random
from torch.distributions import Categorical
import numpy as np
import RoboMaster
import os
import DFSM
import FSM
import math
import copy
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_ob():
    ob = []
    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.random()*2*math.pi)
    ob.append(random.randint(0,1200))
    ob.append(random.randint(0,40))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.random()*2*math.pi)
    ob.append(random.randint(0,2600))
    ob.append(random.randint(0,350))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))
    
    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))

    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.randint(0,1))

    is_ob = random.randint(0,1)

    ob = np.array(ob)
    return ob,is_ob


def beh_state(ob,is_ob):
    s = {"shoot":0,"chase":1,"escape":2, "addblood":3, "addbullet":4}
    self_bullet = ob[4]
    self_hp = ob[3]
    bullet_state = ob[21]
    blood_state =ob[18]
    self_x = int(ob[0])
    self_y = int(ob[1])
    enemy_x = int(ob[5])
    enemy_y = int(ob[6])

    if self_bullet<30 and bullet_state:
        return s['addbullet']
    else:
        if self_hp<1000 and blood_state:
            return s['addblood']
        else:
            if self_bullet>10 and self_hp>500:
                if not is_ob:
                    return s['shoot']
                else:
                    return s['chase']

            else:
                return s['escape']

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=1024,fsmstate_dim=5):
        super(ActorCritic, self).__init__()
        self.fsmstate_dim = fsmstate_dim
        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim+fsmstate_dim+1, 4096),
            nn.ELU(),
            nn.Linear(4096,4096),
            nn.ELU(),
            nn.Linear(4096,2048),
            nn.ELU(),
            nn.Linear(2048,1024),
            nn.ELU(),
            nn.Linear(1024, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim+fsmstate_dim+1, 4096),
            nn.Sigmoid(),
            nn.Linear(4096,2048),
            nn.Sigmoid(),
            nn.Linear(2048,2048),
            nn.Sigmoid(),
            nn.Linear(2048,1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1)
        )

    def nomalization(self,ob_input):
        ob = copy.deepcopy(ob_input)
        ob[0] = ob[0]/808
        ob[1] = ob[1]/448
        ob[2] = ob[2]/(2*math.pi)
        ob[3] = ob[3]/2600
        ob[4] = ob[4]/350

        ob[5] = ob[5]/808
        ob[6] = ob[6]/448
        ob[7] = ob[7]/(2*math.pi)
        ob[8] = ob[8]/2600
        ob[9] = ob[9]/350

        ob[10] = ob[10]/808
        ob[11] = ob[11]/448

        ob[13] = ob[13]/808
        ob[14] = ob[14]/448

        ob[16] = ob[16]/808
        ob[17] = ob[17]/448

        ob[19] = ob[19]/808
        ob[20] = ob[20]/448

        ob[22] = ob[22]/808
        ob[23] = ob[23]/448

        ob[25] = ob[25]/808
        ob[26] = ob[26]/448
        return ob

    def forward(self,ob,state,ground_truth,is_obs):
        temp = torch.zeros(self.fsmstate_dim)
        temp[ground_truth] = 1
        ground_truth = temp.float().to(device)

        temp = torch.zeros(self.fsmstate_dim+1)
        temp[state] = 1
        temp[self.fsmstate_dim] = float(is_obs)
        fsmstate = temp.float().to(device)
        ob = self.nomalization(ob)
        state = torch.from_numpy(ob).float().to(device)
        state = torch.cat((state,fsmstate),dim=0)
        # state,fsmstate = Variable(state.cuda()),Variable(fsmstate.cuda())
        action_probs = self.action_layer(state)
        # sys.stdout.write("{}/{}".format(action_probs,ground_truth))
        # sys.stdout.flush()
        # 那么问题来了，为什么模型的输出的grad是None？
        # print(action_probs,ground_truth)
        loss = (action_probs- ground_truth )**2
        return loss.mean()

    def act(self, state, fsmstate,memory,is_obs):
        temp = torch.zeros(self.fsmstate_dim+1)
        temp[fsmstate] = 1
        temp[self.fsmstate_dim] = is_obs
        fsmstate = temp.float().to(device)

        state = torch.from_numpy(state).float().to(device)
        state = torch.cat((state,fsmstate),dim=0)
        action_probs = self.action_layer(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        # 返回动作的编号
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy



if __name__ == '__main__':
    network = ActorCritic(state_dim=28,action_dim=5)
    if device.type == 'cuda':
        network.cuda()
    adam = torch.optim.Adam(network.parameters(), lr=0.00001)
    # if os.path.exists('./PPO_blue_ICRA.pth'):
    #     network.load_state_dict(torch.load('./PPO_blue_ICRA.pth', map_location='cpu'))
    #     print("load blue model sucessfully")
    for i in range(1000):
        e_loss = 0
        adam.zero_grad()
        for j in range(1000):
            ob,is_ob = random_ob()
            state_input = random.randint(0,4)
            state_out = beh_state(ob,is_ob)
            loss = network(ob,state_input,state_out,is_ob)
            loss.backward()
            e_loss +=loss
        adam.step()
        print("{}".format(e_loss))
        # print('e:{} loss:{}'.format(i,e_loss))
        if i % 200 == 0:
            # print("i_episode:{}".format(i))
            torch.save(network.state_dict(), './PPO_blue_elu.pth')
