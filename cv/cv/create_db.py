
import argparse
import glob
import os
import torch
from torchvision import transforms
from PIL import Image
from network_db import Vgg16

def createDatabase(paths, gpu):
	# Create model
	model = Vgg16()
	# Set transformation
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	data_preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize])
	# Set model to GPU/CPU
	device = 'cpu'
	if gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(gpu)
	model = model.to(device)
	# Get features
	with torch.no_grad():
		features = torch.cat(
			[model(data_preprocess(Image.open(path, 'r').convert('RGB')).unsqueeze(0).to(device)).to('cpu')
				for path in paths],
			dim = 0
		)
	# Show created dataset size
	print('dataset size : {}'.format(len(features)))
	return features

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(create database)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--dataset', '-d', default='default_dataset_path',
						help='Directory for creating database')
	args = parser.parse_args()

	data_dir = args.dataset


	# Get a list of pictures
	paths = glob.glob(os.path.join(data_dir, './*/*.png'))

	assert len(paths) != 0 
	# Create the database
	features = createDatabase(paths, args.gpu)
	# Save the data of database
	torch.save(features, 'result/feature.pt')
	torch.save(paths, 'result/path.pt')

if __name__ == '__main__':
	main()
