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
			""" 穴埋め　"""
			#データセットを参照して適切な辞書を作成する
		}
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []
		for cls in classes:
			cls_dir = os.path.join(self.root, mode, cls)

			img_path_list = """ 穴埋め　""" #glob,os,pathlibなどのモジュールを使うことが考えられる。その際、cls_dirを用いるとよい。
			cls_label = """ 穴埋め　"""
			cls_data = [[img_path, cls_label] for img_path in img_path_list]

			self.all_data.extend(cls_data)

	def __len__(self):
		#データセットの数を返す関数
		return len(self.all_data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		img = Image.open(""" 穴埋め　""").convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = """ 穴埋め　"""
		return [img, label]