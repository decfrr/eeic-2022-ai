import sys
import torch
from torch import optim

import sentence_data
from sentence_data import EOS_ID
from translator_model import TranslatorModel

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

optimizer = optim.Adam(model.parameters())

epoch_num = 10
for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    sum_loss = 0.0
    for i, (english_sentence, japanese_sentence) in enumerate(
            zip(dataset.english_sentences(), dataset.japanese_sentences())):

        model.reset_state()
        optimizer.zero_grad()

        english_sentence = torch.tensor(english_sentence,dtype=torch.long).unsqueeze(-1)
        japanese_sentence = torch.tensor(japanese_sentence,dtype=torch.long).unsqueeze(-1)
        loss = model(english_sentence, japanese_sentence)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss.data.to('cpu'))

        if (i + 1) % 100 == 0:
            print("{0} / {1} sentences finished.".format(
                i + 1, dataset.sentences_size()))

    print("mean loss = {0}.".format(sum_loss / dataset.sentences_size()))

    # 1 epoch 毎にファイルに書き出す
    model_file = "trained_model/translator_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)
