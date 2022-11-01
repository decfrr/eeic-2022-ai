import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import numpy as np


class ReplayBuffer(object):
    """リプレイバッファ
    
    Experience Replayのために使われる
    """
    def __init__(self, capacity):
        """初期化
        
        Args:
            capacity (int): バッファサイズ
        """
        self.capacity = capacity
        self.memory = []  # これまでの経験を保持する
        self.index = 0  # 保存した経験のindex

    def push(self, state, action, next_state, reward, is_state_terminal):
        """リプレイバッファに観測や行動を格納する
        
        Args:
            state (object): 状態
            action (int): 行動
            next_state (object): 次状態
            reward (float): 報酬
            is_state_terminal (bool): 今の状態が終端状態かを表すbool値
        """
        if len(self.memory) < self.capacity:  # まだバッファに空きがあれば
            self.memory.append(None)
        self.memory[self.index] = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_state_terminal=is_state_terminal
            )
        self.index = (self.index + 1) % self.capacity  # indexを1つ進める。バッファに空きがない場合は古いやつから消していく。

    def sample(self, batch_size):
        """batch_sizeの値だけランダムにリプレイバッファから経験のサンプリングを行う
        
        Args:
            batch_size (int): バッチサイズ
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



