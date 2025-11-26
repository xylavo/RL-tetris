from model_dqn_agent import DQNAgent
import torch
from torch import nn
import random
import numpy as np
import datetime
from utils import create_env
from collections import deque
from configuration import Configuration

if __name__ == "__main__":
    env = create_env()
    dqn_agent = DQNAgent(env, Configuration)
    # dqn_agent.board_cnn.load_state_dict(torch.load("dqn_agent_latest_board_80000.pth"))
    # dqn_agent.next_block_mlp.load_state_dict(torch.load("dqn_agent_latest_next_block_80000.pth"))
    # dqn_agent.fc.load_state_dict(torch.load("dqn_agent_latest_fc_80000.pth"))
    state = env.reset()

    print(env.board[4:,:10])
    print(env.block_shape[env.block[0],0])
    while True:
        board = np.array([[state[0]]])
        block = np.array([state[1]])
        time = np.array([[state[2]]])
        board = torch.FloatTensor(board).float()
        block = torch.FloatTensor(block).float()
        time = torch.FloatTensor(time).float()
        action = dqn_agent.get_argmax_action(board, block, time)
        print(action // 10, action % 10)
        input()
        next_state, reward, done = env.step(action // 10, action % 10)
        state = next_state
        print(env.board[4:,:10])
        print(env.block_shape[env.block[0],0])
        if done:
            break
