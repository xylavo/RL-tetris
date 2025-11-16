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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.next_block_mlp = nn.Sequential(
            nn.Linear(5 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(4800 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.target_board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.target_next_block_mlp = nn.Sequential(
            nn.Linear(5 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.target_fc = nn.Sequential(
            nn.Linear(4800 + 64, 512),
            nn.ReLU(),
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
        self.board_optimizer = torch.optim.AdamW(self.board_cnn.parameters(), lr=self.config.lr, weight_decay=1e-3)
        self.next_block_optimizer = torch.optim.AdamW(self.next_block_mlp.parameters(), lr=self.config.lr, weight_decay=1e-3)
        self.fc_optimizer = torch.optim.AdamW(self.fc.parameters(), lr=self.config.lr, weight_decay=1e-3)
    
    def forward(self, board, next_block):
        board_features = self.board_cnn(board)
        next_block_features = self.next_block_mlp(next_block)
        combined = torch.cat((board_features, next_block_features), dim=1)
        q_values = self.fc(combined)
        return q_values
    
    def forward_target_network(self, board, next_block):
        board_features = self.target_board_cnn(board)
        next_block_features = self.target_next_block_mlp(next_block)
        combined = torch.cat((board_features, next_block_features), dim=1)
        q_values = self.target_fc(combined)
        return q_values
        
if __name__ == "__main__":
    config = Configuration()
    env = create_env()
    agent = DQNAgent(env, config)
    print(agent.shape())  # Print the output shape of the CNN part