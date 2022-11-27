import numpy as np
import pysptk as sptk
import pyworld as pw
from scipy.io import wavfile
import os
import sys
import time
import argparse
from train import *

dim = 25
n_units = 128

model = VCDNN(dim,n_units)
_ = model.load_state_dict(torch.load("model/vcmodel.model"))

# test data
x = []
datalist = []
with open("conf/eval.list","r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)

for d in datalist:
    with open("data/SF/mgc/{}.mgc".format(d),"rb") as f:
        dat = np.fromfile(f,dtype="<f8",sep="")
        x.append(dat.reshape(len(dat)//dim,dim))

if not os.path.isdir("result"):
    os.mkdir("result")
if not os.path.isdir("result/wav"):
    os.mkdir("result/wav")
if not os.path.exists("data/SF-TF"):
    os.mkdir("data/SF-TF")
if not os.path.exists("data/SF-TF/mgc"):
    os.mkdir("data/SF-TF/mgc")

fs = 16000
fftlen = 512
alpha = 0.42
for i in range(0,len(datalist)):
    outfile = "result/wav/{}.wav".format(datalist[i])
    with open("data/SF/f0/{}.f0".format(datalist[i]),"rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
    with open("data/SF/ap/{}.ap".format(datalist[i]),"rb") as f:
        ap = np.fromfile(f, dtype="<f8", sep="")
        ap = ap.reshape(len(ap)//(fftlen+1),fftlen+1)
    y = model.get_predata(x[i])
    y = y.astype(np.float64)
    with open("data/SF-TF/mgc/{}.mgc".format(datalist[i]), "wb") as f:
        y.tofile(f)
    sp = sptk.mc2sp(y, alpha, fftlen*2)
    owav = pw.synthesize(f0, sp, ap, fs)
    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))

os.system("ls result/wav")

