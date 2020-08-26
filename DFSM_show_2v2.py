import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import RoboMaster2v2 as RoboMaster
import os
import DFSM
import FSM
import math
import copy
from diff_active.DFSM_Random_pre import ActorCritic
from diff_active.HDFSM_2v2 import ActorCritic2v2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def main():
    ############## Hyperparameters ##############
    env_name = 'ACML'
    state_show = ['shoot','chase','escape','addblood','addbullet']
    # creating environment
    DeepFSM = ActorCritic(state_dim=28,action_dim=5)
    HDFSM = ActorCritic2v2(state_dim=38,fsmstate_dim=2)

    if os.path.exists('./PPO_blue_ACML.pth'):
        DeepFSM.load_state_dict(torch.load('./PPO_blue_ACML.pth', map_location='cpu'))
        print("load blue model sucessfully")

    if os.path.exists('./PPO_blue_HDFSM2v2.pth'):
        HDFSM.load_state_dict(torch.load('./PPO_blue_HDFSM2v2.pth', map_location='cpu'))
        print("load HDFSM blue model sucessfully")

    env = RoboMaster.RMEnv()
    state_dim = 28
    action_dim = 5
    render = True
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 3600  # max timesteps in one episode #3600
    n_latent_var = 1024  # number of variables in hidden layer
    update_timestep = 300  # update policy every n timesteps #18000
    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    # blue_ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)


    # if os.path.exists('./PPO_blue_ACML.pth'):
    #     blue_ppo.policy.load_state_dict(torch.load('./PPO_blue_ACML.pth', map_location='cpu'))
    #     blue_ppo.policy_old = blue_ppo.policy
    #     print("load blue model sucessfully")

    DFSM.prm.createPRM()
    FSM.prm.createPRM()
    # logging variables
    timestep = 0

    mem_reward = 0
    blue_win = 0
    # training loop
    for i_episode in range(1, max_episodes + 1):
        DFSM.prm.createPRM()
        FSM.prm.createPRM()
        ob_numpy = env.reset()
        blue_running_reward = 0
        ob = list(ob_numpy)
        hdfsm_state = HDFSM.act(ob_numpy,0)
        if hdfsm_state == 0:
            blue1ob = ob[0:5]+ob[10:15]+ob[20:]
            blue2ob = ob[5:10]+ob[10:15]+ob[20:]
        else:
            blue1ob = ob[0:5]+ob[15:20]+ob[20:]
            blue2ob = ob[5:10]+ob[15:20]+ob[20:]

        red1ob = ob[0:5] + ob[10:15] + ob[20:]
        red2ob = ob[5:10] + ob[15:20] + ob[20:]

        blue_st1 = DFSM.Statement(blue1ob)
        blue_st2 = DFSM.Statement(blue2ob)
        red_st1 = FSM.Statement(red1ob)
        red_st2 = FSM.Statement(red2ob)

        red_running_reward=0
        for t in range(max_timesteps):
            timestep += 1

            action_blue1, is_update = blue_st1.run(blue1ob, choose='beh')
            action_blue2,_ = blue_st2.run(blue2ob, choose='beh')
            action_red1 = red_st1.run(red1ob, timestep)
            action_red2 = red_st2.run(red2ob, timestep)


            # action_red = red_ppo.policy_old.act(state,red_ppo.memory)
            ob_numpy, reward, done, _ = env.step((action_blue1,action_blue2,action_red1,action_red2))
            ob = list(ob_numpy)
            hdfsm_state = HDFSM.act(ob_numpy,hdfsm_state)
            if hdfsm_state== 0:
                blue1ob = ob[0:5]+ob[10:15]+ob[20:]
                blue2ob = ob[5:10]+ob[10:15]+ob[20:]
            else:
                blue1ob = ob[0:5]+ob[15:20]+ob[20:]
                blue2ob = ob[5:10]+ob[15:20]+ob[20:]
            red1ob = ob[0:5] + ob[10:15] + ob[20:]
            red2ob = ob[5:10] + ob[15:20] + ob[20:]

            if render:
                env.render()
            if done:
                break

        if ob[3]+ob[8] > ob[13]+ob[18]:
            print('blue win')
        else:
            print('red win')

if __name__ == '__main__':
    main()