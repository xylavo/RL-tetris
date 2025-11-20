import torch
from torch import nn
import random
import numpy as np
import datetime
from utils import create_env
from collections import deque
from configuration import Configuration

class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.buffer = deque([],maxlen=self.config.replay_capacity)
    
    def getsize(self):
        return len(self.buffer)
    
    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, size):
        buffer_size = len(self.buffer)
        if size <= buffer_size:
            samples = random.sample(self.buffer, size)
        else:
            assert False, "Not enough samples in replay buffer"
        
        return samples
    
class DQNAgent(nn.Module):
    def __init__(self, env, config):
        super(DQNAgent, self).__init__()
        self.config = config
        self.replay_memory = ReplayMemory(self.config)
        self.num_actions = 40

        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SELU(),
            nn.Flatten()
        )
        # next block MLP + time for target network
        self.next_block_mlp = nn.Sequential(
            nn.Linear(5 * 7 + 1, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(4800 + 64, 512),
            nn.SELU(),
            nn.Linear(512, self.num_actions)
        )

        self.target_board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.SELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SELU(),
            nn.Flatten()
        )
        # next block MLP + time for target network
        self.target_next_block_mlp = nn.Sequential(
            nn.Linear(5 * 7 + 1, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU()
        )
        self.target_fc = nn.Sequential(
            nn.Linear(4800 + 64, 512),
            nn.SELU(),
            nn.Linear(512, self.num_actions)
        )

        for param in self.target_board_cnn.parameters():
            param.requires_grad = False
        for param in self.target_next_block_mlp.parameters():
            param.requires_grad = False
        for param in self.target_fc.parameters():
            param.requires_grad = False
    
    def update_target_network(self):
        self.target_board_cnn.load_state_dict(self.board_cnn.state_dict())
        self.target_next_block_mlp.load_state_dict(self.next_block_mlp.state_dict())
        self.target_fc.load_state_dict(self.fc.state_dict())
    
    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.board_cnn.parameters()) +
            list(self.next_block_mlp.parameters()) +
            list(self.fc.parameters()),
            lr=self.config.lr,
            weight_decay=1e-3
        )
    
    def forward(self, board, next_block, time):
        board_features = self.board_cnn(board)
        next_block_features = self.next_block_mlp(torch.cat((next_block, time), dim=1))
        combined = torch.cat((board_features, next_block_features), dim=1)
        q_values = self.fc(combined)
        return q_values
    
    def forward_target_network(self, board, next_block, time):
        board_features = self.target_board_cnn(board)
        next_block_features = self.target_next_block_mlp(torch.cat((next_block, time), dim=1))
        combined = torch.cat((board_features, next_block_features), dim=1)
        q_values = self.target_fc(combined)
        return q_values

    def get_argmax_action(self, board, next_block, time):
        q_values = self.forward(board, next_block, time)
        action = q_values.argmax(dim=1).item()
        return action
    
    def train(self):
        transitions = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_board = np.expand_dims(np.stack([s[0] for s in states], axis=0),1) # shape: (batch_size, 1, board_height, board_width)
        states_next_block = np.stack([s[1] for s in states], axis=0) # shape: (batch_size, next_block_feature_size)
        states_time = np.expand_dims(np.stack([s[2] for s in states], axis=0),-1) # shape: (batch_size, 1)
        actions = np.stack(actions, axis=0, dtype=np.int64) # shape: (batch_size)
        rewards = np.stack(rewards, axis=0) # shape: (batch_size)
        next_states_board = np.expand_dims(np.stack([s[0] for s in next_states], axis=0),1) # shape: (batch_size, 1, board_height, board_width)   
        next_states_next_block = np.stack([s[1] for s in next_states], axis=0) # shape: (batch_size, next_block_feature_size)
        next_states_time = np.expand_dims(np.stack([s[2] for s in next_states], axis=0),-1) # shape: (batch_size, 1)
        dones = np.stack(dones, axis=0) # shape: (batch_size)

        states_board_tensor = torch.tensor(states_board).float()
        states_next_block_tensor = torch.tensor(states_next_block).float()
        states_time_tensor = torch.tensor(states_time).float()
        actions_tensor = torch.tensor(actions)
        rewards_tensor = torch.tensor(rewards).float()
        next_states_board_tensor = torch.tensor(next_states_board).float()
        next_states_next_block_tensor = torch.tensor(next_states_next_block).float()
        next_states_time_tensor = torch.tensor(next_states_time).float()
        dones_tensor = torch.tensor(dones).float()

        q_values = self.forward(states_board_tensor, states_next_block_tensor, states_time_tensor)
        next_q_values = self.forward_target_network(next_states_board_tensor, next_states_next_block_tensor, next_states_time_tensor)

        chosen_q_values = q_values.gather(dim=-1, index=actions_tensor.reshape(-1,1)).reshape(-1)
        target_q_values = rewards_tensor + self.config.gamma * (1 - dones_tensor) * next_q_values.max(dim=-1).values

        criterion = nn.SmoothL1Loss()
        loss = criterion(chosen_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
def get_eps(config, step):
    eps_init = config.eps_init
    eps_final = config.eps_final
    if step >= config.eps_decrease_step:
        return eps_final
    else:
        m = (eps_final - eps_init) / config.eps_decrease_step
        eps = eps_init + m * step
        return eps
    
def eval_agent(config, env, agent):
    score_sum = 0
    step_count_sum = 0
    for _ in range(config.num_eval_episodes):
        state = env.reset()
        done = False
        step_count = 0
        score = 0
        while not done:
            with torch.no_grad():
                board = torch.tensor(state[0]).unsqueeze(0).unsqueeze(0).float()
                next_block = torch.tensor(state[1]).unsqueeze(0).float()
                time = torch.tensor([[state[2]]]).float()
                action = agent.get_argmax_action(board, next_block, time)
            
            next_state, reward, done = env.step(action // 10, action % 10)
            step_count += 1
            score += reward
            state = next_state
        score_sum += score
        step_count_sum += step_count

    score_evg = score_sum / config.num_eval_episodes
    step_count_evg = step_count_sum / config.num_eval_episodes
    return score_evg, step_count_evg
        
if __name__ == "__main__":
    env = create_env()
    env_eval = create_env()
    agent = DQNAgent(env, Configuration)
    agent.set_optimizer()

    init_replay_buffer_size = int(Configuration.replay_capacity * Configuration.replay_init_ratio)
    state = env.reset()
    step_count = 0
    for _ in range(init_replay_buffer_size):
        a = np.random.choice(40)
        next_state, reward, done = env.step(a // 10, a % 10)
        step_count += 1

        transition = (state, a, reward, next_state, done)
        agent.replay_memory.append(transition)

        state = next_state
        if done:
            state = env.reset()
            step_count = 0
    
    s=env.reset()
    step_count = 0
    for step_train in range(Configuration.train_env_steps):
        eps = get_eps(Configuration, step_train)
        if np.random.rand() < eps:
            a = np.random.choice(40)
        else:
            with torch.no_grad():
                board = torch.tensor(s[0]).unsqueeze(0).unsqueeze(0).float()
                next_block = torch.tensor(s[1]).unsqueeze(0).float()
                time = torch.tensor([[s[2]]]).float()
                a = agent.get_argmax_action(board, next_block, time)
        
        next_s, reward, done = env.step(a // 10, a % 10)
        step_count += 1

        transition = (s, a, reward, next_s, done)
        agent.replay_memory.append(transition)

        s = next_s
        if done:
            s = env.reset()
            step_count = 0

        if step_train % Configuration.target_update_period == 0:
            agent.update_target_network()
        
        if step_train % 4 == 0:
            loss = agent.train()

        if step_train % Configuration.eval_period == 0:
            score_evg, step_count_evg = eval_agent(Configuration, env_eval, agent)
            print(f"Step: {step_train}, Eval Score Evg: {score_evg}, Eval Step Count Evg: {step_count_evg}")
            torch.save(agent.target_board_cnn.state_dict(), f"dqn_agent_latest_board_{step_train}.pth")
            torch.save(agent.target_next_block_mlp.state_dict(), f"dqn_agent_latest_next_block_{step_train}.pth")
            torch.save(agent.target_fc.state_dict(), f"dqn_agent_latest_fc_{step_train}.pth")

    # torch.save(agent.state_dict(), f"dqn_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    
