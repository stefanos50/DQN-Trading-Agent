import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim, nn
from DQN import DQN
import torch.nn.functional as F

class Agent:
    def __init__(self, input_size, output_size, device='cpu', learning_rate= 0.001, gamma=0.99, epsilon=0.6, epsilon_min=0.01, epsilon_decay=0.9995,batch_size=32,memory_size=100):
        self.device = device
        self.output_size = output_size
        self.policy_net = DQN(input_size, output_size).to(device)
        self.target_net = DQN(input_size, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=0.0001)
        self.memory = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lossfn = nn.MSELoss()
        self.history = {'loss':[]}

    def make_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.output_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            best_action, best_action_index = torch.max(q_values[0], 1)
            action = best_action_index.item()
            return action

    def make_eval_action(self,state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            best_action, best_action_index = torch.max(q_values[0], 1)
            action = best_action_index.item()
            return action

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def split_batch(self,batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for experience in batch:
            states.append(self.memory[experience][0])
            actions.append(self.memory[experience][1])
            rewards.append(self.memory[experience][2])
            next_states.append(self.memory[experience][3])
            dones.append(self.memory[experience][4])
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)


    def update_policy(self):
        self.policy_net.train()
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)

        states,actions,rewards,next_states,dones = self.split_batch(batch)

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(state_batch).squeeze(1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).squeeze(1)
            next_q_values,_ = torch.max(next_q_values,1)
        expected_q_values = (next_q_values * self.gamma) * (1 - done_batch) + reward_batch

        loss = self.lossfn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.history['loss'].append(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def store_transition(self, transition):
        self.memory.append(transition)

    def __len__(self):
        return len(self.memory)

    def save_model(self):
        self.policy_net.save_model()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_mode_eval(self):
        self.policy_net.eval()

    def set_mode_train(self):
        self.policy_net.train()