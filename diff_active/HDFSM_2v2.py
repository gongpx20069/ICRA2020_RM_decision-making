import torch
import torch.nn as nn
import random
from torch.distributions import Categorical
import numpy as np
import os
import math
import copy
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_ob():
    ob = []
    # 蓝车1
    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.random()*2*math.pi)
    ob.append(random.randint(0,2600)) # HP max2600
    ob.append(random.randint(0,350)) # 子弹 max350
    # 蓝车2
    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.random()*2*math.pi)
    ob.append(random.randint(0,2600))
    ob.append(random.randint(0,350))
    # 红车1
    ob.append(random.randint(0,808))
    ob.append(random.randint(0,448))
    ob.append(random.random()*2*math.pi)
    ob.append(random.randint(0,2600))
    ob.append(random.randint(0,350))
    # 红车2
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

    # is_ob = random.randint(0,1)

    ob = np.array(ob)
    return ob


def WSM(HP, Bullets, alpha=0.7):
    return alpha*HP/2600 + (1-alpha)*Bullets/250

def wsm_state(ob):
    red1wsm = WSM(ob[13],ob[14])
    red2wsm = WSM(ob[18],ob[19])
    if red1wsm > red2wsm:
        return 1
    else:
        return 0


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


class ActorCritic2v2(nn.Module):
    def __init__(self, state_dim, n_latent_var=1024,fsmstate_dim=5):
        super(ActorCritic2v2, self).__init__()
        self.fsmstate_dim = fsmstate_dim
        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim+fsmstate_dim, 4096),
            nn.Hardtanh(),
            nn.Linear(4096,4096),
            nn.Hardtanh(),
            nn.Linear(4096,2048),
            nn.Hardtanh(),
            nn.Linear(2048,1024),
            nn.Hardtanh(),
            nn.Linear(1024, self.fsmstate_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim+fsmstate_dim, 4096),
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
        ob[12] = ob[12]/(2*math.pi)
        ob[13] = ob[13]/2600
        ob[14] = ob[14]/350

        ob[15] = ob[15]/808
        ob[16] = ob[16]/448
        ob[17] = ob[17]/(2*math.pi)
        ob[18] = ob[18]/2600
        ob[19] = ob[19]/350

        ob[20] = ob[20]/808
        ob[21] = ob[21]/448

        ob[23] = ob[23]/808
        ob[24] = ob[24]/448

        ob[26] = ob[26]/808
        ob[27] = ob[27]/448

        ob[29] = ob[29]/808
        ob[30] = ob[30]/448

        ob[32] = ob[32]/808
        ob[33] = ob[33]/448

        ob[35] = ob[35]/808
        ob[36] = ob[36]/448
        return ob

    def forward(self,ob,state,ground_truth):
        temp = torch.zeros(self.fsmstate_dim)
        temp[ground_truth] = 1
        ground_truth = temp.float().to(device)

        temp = torch.zeros(self.fsmstate_dim)
        temp[state] = 1
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

    def act(self, state, fsmstate,memory=None):
        temp = torch.zeros(self.fsmstate_dim)
        temp[fsmstate] = 1
        fsmstate = temp.float().to(device)

        state = torch.from_numpy(state).float().to(device)
        state = torch.cat((state,fsmstate),dim=0)
        action_probs = self.action_layer(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        # 返回动作的编号
        action = dist.sample()

        if memory != None:
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
    network = ActorCritic2v2(state_dim=38,fsmstate_dim=2)
    if device.type == 'cuda':
        network.cuda()
    adam = torch.optim.Adam(network.parameters(), lr=0.000001)
    if os.path.exists('./PPO_blue_HDFSM2v2.pth'):
        network.load_state_dict(torch.load('./PPO_blue_HDFSM2v2.pth', map_location='cpu'))
        print("load blue HDFSM model sucessfully")
    for i in range(1000):
        e_loss = 0
        adam.zero_grad()
        for j in range(1000):
            ob = random_ob()
            state_input = random.randint(0,1)
            state_out = wsm_state(ob)
            loss = network(ob,state_input,state_out)
            loss.backward()
            e_loss +=loss
        adam.step()
        print("{}".format(e_loss))
        # print('e:{} loss:{}'.format(i,e_loss))
        if i % 20 == 0:
            print("i_episode:{}".format(i))
            torch.save(network.state_dict(), './PPO_blue_HDFSM2v2.pth')
