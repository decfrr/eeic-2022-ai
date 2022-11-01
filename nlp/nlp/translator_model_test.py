import sys
import torch

import sentence_data
from translator_model import TranslatorModel

# dataset = sentence_data.SentenceData("dataset/data_1000.txt")
dataset = sentence_data.SentenceData("dataset/data_full.txt.txt")

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

model.load_state_dict(torch.load("trained_model/translator_10.model"))

# 入力された文章を単語に分割する
sentence = input("input an english sentence : ").split(' ')
# 単語IDのリストに変換する
sentence_id = []
for word in sentence:
    if not word:
        # 単語が空だったら飛ばす
        continue
    word = word.lower()
    id = dataset.english_word_id(word)
    if id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    else:
        id = torch.tensor(id,dtype=torch.long).unsqueeze(-1)
        sentence_id.append(id)

japanese = model(torch.stack(sentence_id))
for id in japanese:
    print(dataset.japanese_word(id), end='')
print()
