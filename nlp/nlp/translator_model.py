import torch
import torch.nn as nn
from torch import cuda      

from sentence_data import EOS_ID

# GPU上で計算を行う場合は，この変数を非Noneの整数、又は"cuda"にする
gpu_id = None

# Encoder-Decoderモデルを用いた翻訳モデルの定義
class TranslatorModel(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, embed_size=100):
        super(TranslatorModel, self).__init__()
        self.W_x_hi=nn.Embedding(source_vocabulary_size, embed_size)
        self.W_y_hi=nn.Embedding(target_vocabulary_size, embed_size)
        self.W_lstm_enc=nn.LSTMCell(embed_size, embed_size)
        self.W_lstm_dec=nn.LSTMCell(embed_size, embed_size)
        self.W_hr_y=nn.Linear(embed_size, target_vocabulary_size)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # 隠れ層の次元数を保存
        self.embed_size = embed_size
        self.reset_state()

        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')

    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.hr = None

    def encode(self, word):
        hi = self.W_x_hi(word)
        self.hr = self.W_lstm_enc(hi,self.hr)

    def decode(self, hi):
        self.hr = self.W_lstm_dec(hi,self.hr)
        y = self.W_hr_y(self.hr[0])
        return y
    
    # (target_words is not None)入力データと正解データから，lossを計算する
    # (target_words is None)又は現在のモデルを用いて，入力単語列から，翻訳結果の出力単語列を返す
    def forward(self, source_words, target_words=None):
        source_words = source_words.to(self.device)
        if target_words is not None:
            target_words = target_words.to(self.device)
        for word in source_words:
            self.encode(word)

        # メモリーセルの状態は引き継がない
        self.hr = (self.hr[0], torch.zeros(1, self.embed_size).to(self.device))
        eos_token = torch.tensor(EOS_ID).view(1, 1).to(self.device)
        hi = self.W_x_hi(eos_token.squeeze(0))

        if target_words is not None:
            accum_loss = 0
            for target_word in torch.cat((target_words, eos_token)):
                y = self.decode(hi)
                # 正解の単語とのクロスエントロピーを取ってlossとする
                loss = self.loss_fn(y,target_word)
                accum_loss = accum_loss + loss
                # 正解データをLSTMの入力にする
                hi = self.W_y_hi(target_word)
            return accum_loss
        else:
            result = []
            # 最大30単語で打ち切る
            for _ in range(30):
                y = self.decode(hi)
                # もっともそれらしい単語を得る
                id = y.argmax()
                # EOSが出力されたら打ち切る
                if id == EOS_ID:
                    break
                result.append(id)
                id = id.unsqueeze(-1)
                # 推定した単語をLSTMの入力にする
                hi = self.W_y_hi(id)
            return result
