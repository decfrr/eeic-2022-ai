import os
import sys
import array

from dtw import dtw
import numpy as np
import pysptk as sptk

srcspk = "SF"
tgtspk = "TF"

mgclist = os.listdir("data/{}/mgc".format(srcspk))

if not os.path.isdir("data/{}/data".format(srcspk)):
    os.mkdir("data/{}/data".format(srcspk))
if not os.path.isdir("data/{}/data".format(tgtspk)):
    os.mkdir("data/{}/data".format(tgtspk))

def distfunc(x,y):
    # Euclid distance except first dim
    return np.linalg.norm(x[1:]-y[1:])

dim = 25 # mgc dim + 1
for mf in mgclist:
    print(mf)
    bn, _ = os.path.splitext(mf)
    srcfile = "data/{}/mgc/{}".format(srcspk,mf)
    tgtfile = "data/{}/mgc/{}".format(tgtspk,mf)

    with open(srcfile,"rb") as f:
        x = np.fromfile(f, dtype="<f8", sep="")
        x = x.reshape(len(x)//dim,dim)
    with open(tgtfile,"rb") as f:
        y = np.fromfile(f, dtype="<f8", sep="")
        y = y.reshape(len(y)//dim,dim)
    print("framelen: (x,y) = {} {}".format(len(x),len(y)))
    _,_,_, twf = dtw(x,y,distfunc)
    srcout = "data/{}/data/{}.dat".format(srcspk,bn)
    tgtout = "data/{}/data/{}.dat".format(tgtspk,bn)

    with open(srcout,"wb") as f:
        x[twf[0]].tofile(f)
    with open(tgtout,"wb") as f:
        y[twf[1]].tofile(f)

## training of acoustic model

# Listing training/evaluation data
os.system("mkdir -p conf")
os.system("ls data/SF/data/ | head -45 | sed -e 's/\.dat//' > conf/train.list")
os.system("ls data/SF/data/ | tail -5 | sed -e 's/\.dat//' > conf/eval.list")
