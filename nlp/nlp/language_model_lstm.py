import torch
import torch.nn as nn
from torch import cuda      

# GPU上で計算を行う場合は，この変数を非Noneの整数、又は"cuda"にする
gpu_id = None

# 言語モデル用ニューラルネットワークの定義
class LanguageModelLSTM(nn.Module):
    def __init__(self, source_vocabulary_size, embed_size=100):
        # パラメータを chainer.Chain に渡す
        super(LanguageModelLSTM, self).__init__()
        self.W_x_hi = nn.Embedding(source_vocabulary_size, embed_size)
        self.W_lstm=nn.LSTMCell(embed_size, embed_size)
        # ここのhrはタプル(hidden state, cell state)
        self.hr = None
        self.W_hr_y=nn.Linear(embed_size, source_vocabulary_size)
        self.reset_state()

        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')

    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.hr = None

    def forward(self, word):
        if gpu_id is not None:
            word = word.to(self.device)
        # ここを実装する
        self.hr = self.W_lstm(self.W_x_hi(word), self.hr)
        return self.W_hr_y(self.hr[0])
