import numpy as np
import pysptk as sptk
from scipy.io import wavfile
import os
from pysptk.synthesis import MLSADF, Synthesizer

fs = 16000
fftlen = 512
alpha = 0.42
dim = 25

datalist = []
with open("conf/eval.list", "r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)

for i in range(0,len(datalist)):
    outfile = "result/wav/{}_diff.wav".format(datalist[i])
    with open("data/SF-TF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        conv_mgc = np.fromfile(f, dtype="<f8", sep="")
        conv_mgc = conv_mgc.reshape(len(conv_mgc)//dim, dim)

    with open("data/SF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        src_mgc = np.fromfile(f, dtype="<f8", sep="")
        src_mgc = src_mgc.reshape(len(src_mgc)//dim, dim)
    
    fs, data = wavfile.read("data/SF/wav/{}.wav".format(datalist[i]))  # 入力音声そのものをもってくる
    data = data.astype(np.float)

    diff_mgc = conv_mgc - src_mgc  # 差分のフィルタを用意する

    # 差分のフィルタを入力音声波形に適用する
    b = np.apply_along_axis(sptk.mc2b, 1, diff_mgc, alpha)
    synthesizer = Synthesizer(MLSADF(order=dim-1, alpha=alpha), 80)
    owav = synthesizer.synthesis(data, b)
    
    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))
