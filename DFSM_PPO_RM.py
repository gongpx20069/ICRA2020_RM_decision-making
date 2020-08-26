import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import RoboMaster
import os
import DFSM
import FSM
import math
import copy
from diff_active.DFSM_Random_pre import ActorCritic
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



class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())#先让old=new

        self.MseLoss = nn.MSELoss()
        self.memory = Memory()

    def update(self):
        if len(self.memory.rewards)==0:
            return
        # Monte Carlo estimate of state rewards:
        rewards = []
        # print(len(self.memory.rewards),len(self.memory.states))
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            # print(rewards.size(),state_values.size())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = 'ACML'
    state_show = ['shoot','chase','escape','addblood','addbullet']
    # creating environment
    env = RoboMaster.RMEnv()
    state_dim = 28
    action_dim = 5
    render = False
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
    update_count = 0
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    # memory = Memory()
    blue_ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    # red_ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    # print(lr, betas)


    if os.path.exists('./PPO_blue_pre.pth'):
        blue_ppo.policy.load_state_dict(torch.load('./PPO_blue_pre.pth', map_location='cpu'))
        blue_ppo.policy_old = blue_ppo.policy
        print("load blue model sucessfully")
    # if os.path.exists('./PPO_red_{}.pth'.format(env_name)):
    #     red_ppo.policy.load_state_dict(torch.load('./PPO_red_{}.pth'.format(env_name), map_location='cpu'))
    #     red_ppo.policy_old = red_ppo.policy
    #     print("load red model sucessfully")

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
        ob = env.reset()
        blue_running_reward = 0
        st1 = DFSM.Statement(ob)
        st2 = FSM.Statement(ob)

        red_running_reward=0
        for t in range(max_timesteps):
            # env.render()
            timestep += 1
            # Running policy_old:
            # state = state.reshape((1,10))
            # action_blue = blue_ppo.policy_old.act(state,,blue_ppo.memory)
            action_blue,is_update = st1.run(ob,blue_ppo,choose='dfsm')
            action_red=st2.run(ob, timestep)
            # action_red = red_ppo.policy_old.act(state,red_ppo.memory)
            ob, reward, done, _ = env.step((action_blue,action_red))

            # Saving reward and is_terminal:
            mem_reward += reward[0]
            blue_running_reward += reward[0]
            red_running_reward +=reward[1]
            if is_update:
                # print('blue state {}'.format(state_show[st1.state]))
                blue_ppo.memory.rewards.append(mem_reward)
                blue_ppo.memory.is_terminals.append(done)
                mem_reward = 0
                update_count +=1
                if update_count%update_timestep == 0:
                    print('Episode {} \t mem_reward:{} \t blue_win:{:.2f}'.format(i_episode, sum(blue_ppo.memory.rewards),blue_win/i_episode)) 
                    blue_ppo.update()
                    blue_ppo.memory.clear_memory()
                    update_count = 0

                # print('blue state {}'.format(state_show[st1.state]))
                # print('red state {}'.format(state_show[st2.state]))

            if render:
                env.render()
            if done:
                break

        if ob[3] > ob[8]:
            blue_win+=1       
        # stop training if avg_reward > solved_reward
        if i_episode % 1000 == 0:
            print("i_episode:{}".format(i_episode))
            torch.save(blue_ppo.policy.state_dict(), './PPO_blue_{}.pth'.format(env_name))
            # torch.save(red_ppo.policy.state_dict(), './PPO_red_{}.pth'.format(env_name))
        #     break

        # logging
        # if i_episode % log_interval == 0:
        #     avg_length = int(avg_length / log_interval)
        #     running_reward = int((blue_running_reward / log_interval))

        #     print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        #     running_reward = 0
        #     avg_length = 0


if __name__ == '__main__':
    main()