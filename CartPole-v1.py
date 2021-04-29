import random
import numpy as np
from collections import deque
import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env=gym.make('CartPole-v1')
action_size= env.action_space.n
state_size=env.observation_space.shape[0]
batch_size=32
episodes=1000


class DQNModel(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQNModel, self).__init__()
        self.a1 = nn.Linear(state_size, 24)
        self.a2 = nn.Linear(24, 24)
        self.a3 = nn.Linear(24, action_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.a1(x))
        x = self.relu2(self.a2(x))
        x = (self.a3(x))
        return x

    def save(self, file_name='model.pth'):
        path = './model'
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)


class DQNAgent:
    def __init__(self,state_size,action_size):
        self.memory = deque(maxlen=5000)
        self.learning_rate=0.001
        self.epsilon=1
        self.max_eps=1
        self.min_eps=0.01
        self.eps_decay = 0.005
        self.gamma=0.95
        self.state_size= state_size
        self.action_size= action_size
        self.epsilon_lst=[]
        self.model=DQNModel(state_size, action_size)
        self.criterion= nn.MSELoss()
        self.optimizer= optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def add_memory(self, new_state, reward, done, state, action):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        self.memory.append((new_state, reward, done, state, action))

    def action(self, state):
        if np.random.rand()<= self.epsilon:
            return random.randrange(self.action_size)

        state1= torch.tensor(state, dtype=torch.float)
        act_values= self.model(state1)
        return torch.argmax(act_values).item()

    def replay(self, batch_size, episode):
        minibatch= random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target= reward
            if not done:
                target=reward + self.gamma* torch.max(self.model(new_state)[0])

            pred = self.model(state)
            target_f = pred.clone()
            target_f[0][action.numpy()] = target

            loss = self.criterion(  target_f, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon> self.min_eps:
            self.epsilon= (self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

    def save(self):
        self.model.save()

agent=DQNAgent(state_size, action_size)

done= False
for e in range(episodes):
    state= env.reset()
    state= np.reshape(state, [1, state_size])
    time=0
    while True:
        time += 1
        env.render()
        action = agent.action(state)
        new_state, reward, done, info = env.step(action)
        reward=reward if not done else -10
        new_state= np.reshape(new_state, [1, state_size])
        agent.add_memory(new_state, reward, done, state, action)
        state= new_state

        if done:
            print(f'Episode: {e}/{episodes}. score {time}, e: {float(agent.epsilon):.2}')
            break

    if len(agent.memory)> (batch_size):
        agent.replay(batch_size, e)

    agent.save()
