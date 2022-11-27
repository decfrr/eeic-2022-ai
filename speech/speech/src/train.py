import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
import sys
import time
import argparse

def get_dataset(dim=25):
    x = []
    y = []
    datalist = []
    with open("conf/train.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        print(d)
        with open("data/SF/data/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(dat.reshape(len(dat)//dim,dim))
        with open("data/TF/data/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            y.append(dat.reshape(len(dat)//dim,dim))
    return x,y

class VCDNN(nn.Module):
        def __init__(self, dim=25, n_units=256):
            super(VCDNN, self).__init__()
            self.fc = nn.ModuleList([
                           nn.Linear(dim, n_units),
                           nn.Linear(n_units, n_units),
                           nn.Linear(n_units, dim)
            ])

        def forward(self, x):
            h1 = F.relu(self.fc[0](x))
            h2 = F.relu(self.fc[1](h1))
            h3 = self.fc[2](h2)
            return h3

        def get_predata(self, x):
            _x = torch.from_numpy(x.astype(np.float32))
            return self.forward(_x).detach().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=50,
					    help='number of epochs to train (default: 50)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # Listing training/evaluation data
    os.system("mkdir -p conf")
    os.system("ls data/SF/data/ | head -45 | sed -e 's/\.dat//' > conf/train.list")
    os.system("ls data/SF/data/ | tail -5 | sed -e 's/\.dat//' > conf/eval.list")

    x_train, y_train = get_dataset()
    # parameters for training
    n_epoch = args.epochs
    dim = 25
    n_units = 128
    N = len(x_train)

    model = VCDNN(dim,n_units)
    model.double()
    if args.gpu >= 0:
        device = 'cuda:' + str(args.gpu)
        model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    loss_fn = nn.MSELoss()

    model.train()

    losses = []
    sum_loss = 0

    for epoch in range(1, n_epoch + 1):
        sum_loss = 0

        for i in range(0, N):
            x_batch = torch.from_numpy(x_train[i])
            y_batch = torch.from_numpy(y_train[i])
            if args.gpu >= 0:
                device = 'cuda:' + str(args.gpu)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predict_y_batch = model(x_batch)
            loss = loss_fn(predict_y_batch, y_batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            average_loss = sum_loss / N
            losses.append(average_loss)

            print("epoch: {}/{}  loss: {}".format(epoch, n_epoch, average_loss))

    if not os.path.isdir("model"):
        os.mkdir("model")
    torch.save(model.state_dict(), "model/vcmodel.model")

if __name__ == '__main__':
    main()
