import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
class MyCifarDataset(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = 'train' if train else 'test'
		# TODO class辞書の定義
		classes = {
			#データセットを参照して適切な辞書を作成する
			'airplane':0,
			'automobile':1,
			'bird':2,
			'cat':3,
			'deer':4,
			'dog':5,
			'frog':6,
			'horse':7,
			'ship':8,
			'truck':9

		}
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []
		for cls in classes:
			cls_dir = os.path.join(self.root, mode, cls)
			#glob,os,pathlibなどのモジュールを使うことが考えられる。その際、cls_dirを用いるとよい。
			img_path_list = glob.glob(cls_dir + '/*.png')
			cls_label = classes[cls]
			cls_data = [[img_path, cls_label] for img_path in img_path_list]

			self.all_data.extend(cls_data)

	def __len__(self):
		#データセットの数を返す関数
		return len(self.all_data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		img = Image.open(self.all_data[idx][0]).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = self.all_data[idx][1]
		return [img, label]