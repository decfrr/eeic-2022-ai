from . import models
from .random_agent import RandomAgent
from .rulebase_agent import RulebaseAgent
from .table_q_agent import TableQAgent
from .dqn_agent import DQNAgent

# このように __init__.py で import することで、agentsフォルダ内の構成が変わっても、
# (例えば main.py などの) 外部から
# import agents
# agent = agents.RandomAgent()
# のようにアクセスでき、モジュールとして提供しやすくなる
#
# __init__.py を書かない場合、
# from agents.random_agent import RandomAgent
# agent = RandomAgent()
# のようにアクセスすることになり、外部ファイルにagentsフォルダの中身を直接書くことになる
# 例えば今後 random_agent.py が agents/not_neural_network_agents/random_agent.py に移動したときのことを考えてみるとよい
