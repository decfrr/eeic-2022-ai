
import argparse
import torch
from torchvision import transforms
from PIL import Image
from network_db import Vgg16

def search(src, db_features, db_paths, k, gpu):
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
		src_feature = model(data_preprocess(Image.open(src, 'r').convert('RGB')).unsqueeze(0).to(device)).to('cpu')
	# Load database
	paths = torch.load(db_paths)
	features = torch.load(db_features)
	assert k <= len(paths)
	assert len(features) == len(paths)
	# Calculate distances
	distances = torch.tensor(
		[torch.norm(src_feature - feature)
			for feature in features]
	)
	_, indices = torch.topk(distances, k, largest=False)
	# Show results
	for i in indices:
		print(paths[i])

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Feature extraction(search image)')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--input', '-i', default='default_source_image_path',
						help='path to database features')
	parser.add_argument('--features', '-f', default='result/feature.pt',
						help='path to database features')
	parser.add_argument('--paths', '-p', default='result/path.pt',
						help='path to database paths')
	parser.add_argument('--k', '-k', type=int, default=5,
						help='find num')
	args = parser.parse_args()

	search(args.input, args.features, args.paths, args.k, args.gpu)

if __name__ == '__main__':
	main()