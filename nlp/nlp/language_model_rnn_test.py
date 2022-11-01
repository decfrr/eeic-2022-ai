import sys
import torch

import sentence_data
from sentence_data import EOS_ID
from language_model_rnn import LanguageModelRNN

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = LanguageModelRNN(dataset.japanese_word_size())

model.load_state_dict(torch.load("trained_model/langage_model_rnn_10.model"))
model.eval()

initial_words = input("input initial words : ").split(' ')

next_y = None
for word in initial_words:
    if not word:
        # 単語が空だったら飛ばす
        continue
    print(word)
    word_id = dataset.japanese_word_id(word)
    if word_id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    # 単語をRNNに入力する
    word_id = torch.tensor(word_id,dtype=torch.long)
    next_y = model(word_id)

if next_y is None:
    sys.stderr.write("Error : Empty input\n")
    sys.exit()

# 最大30単語で打ち切る
for i in range(30):
    # もっともそれらしい単語を取ってくる
    word_id = next_y.argmax()
    # EOSが出力されたら終了
    if word_id == EOS_ID:
        sys.exit()
    print(dataset.japanese_word(word_id))
    next_y = model(word_id)
