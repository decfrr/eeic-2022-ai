import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import CifarCNN
from dataset import MyCifarDataset

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	parser.add_argument('--dataset', '-d', default='data/mini_cifar',
						help='Root directory of dataset')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('')

	# Set up a neural network to test
	net = CifarCNN(10)
	# Load designated network weight
	net.load_state_dict(torch.load(args.model))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyCifarDataset(root=args.dataset, train=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
											 shuffle=False, num_workers=2)

	# Test
	correct = 0
	total = 0
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			# Get the inputs; data is a list of [inputs, labels]
			images, labels = data
			if args.gpu >= 0:
				images = images.to(device)
				labels = labels.to(device)
			# Forward
			outputs = net(images)
			# Predict the label
			_, predicted = torch.max(outputs, 1)
			# Check whether estimation is right
			c = (predicted == labels).squeeze()
			for i in range(len(predicted)):
				label = labels[i]
				correct += c[i].item()
				class_correct[label] += c[i].item()
				total += 1
				class_total[label] += 1

	# List of classes
	classes = ('airplane', 'automobile', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	# Show accuracy
	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))
	print('Accuracy : %.3f %%' % (100 * correct / total))


if __name__ == '__main__':
	main()
