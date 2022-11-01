import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents import abstract_agent
import torch.optim as optim
from .utils import Utils
from agents import models
from agents import replay_buffer
import random
import numpy as np


class DQNAgent(abstract_agent.Agent):
    """Deep Q NeuralNetwork Agent
    """
    def __init__(self, env):
        # 観測をいい感じに処理する
        self.obs_space = env.observation_space
        self.flatten, self.obs_size = Utils.get_flatten_function_and_size(self.obs_space)
        self.n_actions = env.action_space.n

        # model (neural network) を作る
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy_net = models.DQNModel(self.obs_size, self.n_actions).to(self.device)
        self.target_net = models.DQNModel(self.obs_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # バッチサイズの設定
        self.batch_size = 32

        # 割引率
        self.discount_factor = 0.95

        # リプレイバッファの設定
        self.replay_buffer_size = 10**6
        self.replay_buffer = replay_buffer.ReplayBuffer(self.replay_buffer_size)

        # epsilon-greedyで使用する
        self.eps = 0.3

        # リプレイバッファに格納された経験がreplay_start_sizeを超えるまでは学習を行わない
        self.replay_start_size = 500

        # target_netとpolicy_netをどれくらいのタイムステップごとに同期させるか
        self.target_update_intervel = 100

        # DQNの更新に使用するオプティマイザ
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)

        # 直前のタイムステップにおける状態と行動
        self.last_state = None
        self.last_action = None

        # 何回更新が行われたか
        self.n_updates = 0

        # 現在のタイムステップ
        self.timestep = 0

    def act_and_train(self, obs, reward, done):
        self.train(obs, reward, done)
        return self.select_action(obs, explore=True)

    def act(self, obs):
        return self.select_action(obs, explore=False)

    def stop_episode_and_train(self, obs, reward, done):
        self.train(obs, reward, done)
        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None

    def save(self, dirname):
        pass

    def load(self, dirname):
        pass

    def get_statistics(self):
        return [('n_updates', self.n_updates)]

    def select_action(self, obs, explore):
        self.timestep += 1
        obs_flatten = self._obs_flatten([obs])

        if explore:
            sample = random.random()
            if sample > self.eps:
                with torch.no_grad():
                    action = self.policy_net(obs_flatten).max(1)[1].item()
            else:
                action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                action = self.policy_net(obs_flatten).max(1)[1].item()

        self.last_action = action
        return action

    def _obs_flatten(self, states):
        features = [self.flatten(s) for s in states]
        features = torch.tensor(features, device=self.device, dtype=torch.float32)
        return features

    def _extract(self, experiences, key):
        experiences_part = []
        if not (key in experiences[0].keys()):
            raise KeyError
        for i in range(self.batch_size):
            experiences_part.append(experiences[i][key])

        return experiences_part

    def train(self, obs, reward, done):
        if done:
            is_state_terminal = True
        else:
            is_state_terminal = False

        # replay_start_size よりリプレイバッファの現在の大きさが小さい場合、訓練を行わない
        if self.replay_start_size >= len(self.replay_buffer):
            if self.last_state is not None:
                # リプレイバッファに今回の記録を格納する
                self.replay_buffer.push(
                    state=self.last_state,
                    action=self.last_action,
                    reward=reward,
                    next_state=obs,
                    is_state_terminal=is_state_terminal
                )
            # 状態の更新
            self.last_state = obs
        else:
            # リプレイバッファから経験をサンプリングする
            # サンプリングしたexperiencesを、扱いやすいようにkeyごとにバッチ化
            experiences = self.replay_buffer.sample(self.batch_size)
            batch_state = self._obs_flatten(self._extract(experiences, 'state'))
            batch_next_state = self._obs_flatten(self._extract(experiences, 'next_state'))
            batch_action = torch.tensor(self._extract(experiences, 'action'), device=self.device, dtype=torch.long).view(-1, 1)
            batch_reward = torch.tensor(self._extract(experiences, 'reward'), device=self.device, dtype=torch.float32).view(-1, 1)
            batch_is_state_terminal = torch.tensor(self._extract(experiences, 'is_state_terminal'), device=self.device, dtype=torch.long).view(-1, 1)

            # 学習を行う
            state_action_values = self.policy_net(batch_state).gather(1, batch_action)  # policy_netで計算した状態のQ値
            with torch.no_grad():
                next_state_values = self.target_net(batch_next_state).max(1)[0].view(-1, 1)  # target_netで計算した次状態のQ値
            batch_mask = (batch_is_state_terminal * -1. + 1.).float()  # 終端状態の場合を識別するマスクの作成
            expected_state_values = (self.discount_factor * next_state_values * batch_mask) + batch_reward  # gamma * Q(s', a') + r の部分 policy_netのパラメータがこれに近づくように学習する

            loss = F.smooth_l1_loss(state_action_values, expected_state_values)  # huber loss の計算

            # モデルの更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.n_updates += 1
        
            if self.last_state is not None:
                # リプレイバッファに今回の記録を格納する
                self.replay_buffer.push(
                    state=self.last_state,
                    action=self.last_action,
                    reward=reward,
                    next_state=obs,
                    is_state_terminal=is_state_terminal
                )

            # 状態の更新
            self.last_state = obs

            # target_netとpolicy_netの同期を行う (fixed target network)
            if self.timestep % self.target_update_intervel == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
