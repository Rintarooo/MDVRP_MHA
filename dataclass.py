import os
import numpy as np
import torch
import math
import json
import sys

from Torch.dataclass import TorchJson
from Ortools.dataclass import OrtoolsJson

from Torch.dataset import generate_data


# only for exporting txt file, not reading
class GAtxt():
	def __init__(self, txt_path):
		self.path = txt_path

	def write_txt(self, src):
		# if isinstance(src['depot_xy'], np.ndarray):
		if isinstance(src['depot_xy'], torch.Tensor):
			data = {}
			for k, v in src.items():
				data[k] = v.squeeze(0).numpy()

		n_car = data['car_capacity'].shape[0]
		n_depot = data['depot_xy'].shape[0]
		n_customer = data['customer_xy'].shape[0]
		with open(self.path, 'w') as f:
			n_car_each_depot = n_car // n_depot
			f.write(f'{n_car_each_depot} {n_customer} {n_depot}\n')
			for _ in range(n_depot):
				val = int(100*data['car_capacity'][0])
				f.write(f'0 {val}\n')	
			for i in range(n_customer):
				x = int(100*data['customer_xy'][i][0])
				y = int(100*data['customer_xy'][i][1])
				d = int(100*data['demand'][i])
				f.write(f'{i+1} {x} {y} 0 {d}\n')
			for i in range(n_depot):	
				x = int(100*data['depot_xy'][i][0])
				y = int(100*data['depot_xy'][i][1])
				f.write(f'{i+1+n_customer} {x} {y}\n') 


if __name__ == '__main__':
	device = torch.device('cpu')# torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
	assert len(sys.argv) >= 2, 'len(sys.argv) should be >= 2'
	print(sys.argv)
	n_depot = int(sys.argv[1])# 3
	n_car_each_depot = int(sys.argv[2])# 5
	n_customer = int(sys.argv[3])# 100
	seed = int(sys.argv[4])# 0
	capa = float(sys.argv[5])# 2.
	
	data = generate_data(device, batch = 1, n_car_each_depot = n_car_each_depot, n_depot = n_depot, n_customer = n_customer, capa = capa, seed = seed)
	
	basename = f'n{n_customer}d{n_depot}c{n_car_each_depot}D{int(capa)}s{seed}.json'
	dirname1 = 'Torch/data/'
	dirname2 = 'Ortools/data/'
	dirname3 = 'GA/data/'
	for x in [dirname1, dirname2, dirname3]:
		os.makedirs(x, exist_ok = True)

	json_path_torch = dirname1 + basename
	json_path_ortools = dirname2 + basename
	txt_path_ga = dirname3 + basename.split('.')[0] + '.txt'
	for x in [json_path_torch, json_path_ortools, txt_path_ga]:
		print(f'generate {x} ...')

	
	hoge1 = TorchJson(json_path_torch)
	hoge1.dump_json(data)
	data = hoge1.load_json(device)
	
	hoge2 = GAtxt(txt_path_ga)
	hoge2.write_txt(data)

	hoge3 = OrtoolsJson(json_path_ortools)
	hoge3.dump_json(data)
	data = hoge3.load_json()