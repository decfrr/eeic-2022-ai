import torch
import torch.nn as nn 
from torch import cuda       

# GPU上で計算を行う場合は，この変数を非Noneの整数、又は"cuda"にする
gpu_id = None

class LanguageModelRNN(nn.Module):
    def __init__(self, source_vocabulary_size, embed_size=100):
        super(LanguageModelRNN, self).__init__()
        self.embed_size = embed_size
        self.W_x_hi = nn.Embedding(source_vocabulary_size, embed_size)
        self.W_hi_hr = nn.Linear(embed_size, embed_size)
        self.W_hr_hr = nn.Linear(embed_size, embed_size)
        self.W_hr_y=nn.Linear(embed_size, source_vocabulary_size)
        
        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.reset_state()

    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.hr = torch.zeros(1, self.embed_size).to(self.device)

    def forward(self, cur_word):
        hi = self.W_x_hi(cur_word.to(self.device))
        self.hr = torch.tanh(self.W_hi_hr(hi)+self.W_hr_hr(self.hr))
        y = self.W_hr_y(self.hr)
        return y
