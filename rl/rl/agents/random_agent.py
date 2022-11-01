import gym
import numpy as np
from agents import abstract_agent


class RandomAgent(abstract_agent.Agent):
    """Random walk agent
    
    離散的な観測・行動空間を持つ環境に対してランダムウォークするagent
    実装を変えれば観測・行動空間が変わってもランダムウォークできるようになるが、
    Rulebase agentと実装を揃えるためにあえて離散性を仮定している

    Args:
        env (gym.Env): 環境（観測・行動の空間を知るために使う）
    """

    def __init__(self, env):
        self.observation_num = {key: val.n for key, val in env.observation_space.spaces.items()}
        self.action_num = env.action_space.n

    def act_and_train(self, obs, reward, done):
        self.train(obs, reward)
        action = self.act(obs)
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.train(obs, reward)
        self.stop_episode()

    def act(self, obs):
        return np.random.randint(self.action_num)

    def train(self, obs, reward):
        pass

    def stop_episode(self):
        pass

    def save(self, dirname):
        pass

    def load(self, dirname):
        pass

    def get_statistics(self):
        return []
