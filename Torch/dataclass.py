import os
import numpy as np
import torch
import math
import json
import sys

class TorchJson():
	def __init__(self, json_path):
		self.json_path = json_path
		
	def dump_json(self, src):
		# https://note.nkmk.me/python-json-load-dump/
		if isinstance(src['depot_xy'], torch.Tensor):
			dst = {}
			for k, v in src.items():
				dst[k] = v.tolist()# since torch tensor can't be convert into json
		with open(self.json_path, 'w') as f:
			json.dump(dst, f, indent = 4)

	def load_json(self, device):
		if not os.path.isfile(self.json_path):
			raise FileNotFoundError
		with open(self.json_path) as f:
			dst = json.load(f)
		for k, v in dst.items():
			dst[k] = torch.tensor(v).to(device)
		return dst
