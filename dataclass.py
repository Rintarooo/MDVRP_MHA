import os
import numpy as np
import torch
import math
import json


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


class OrtoolsJson():
	def __init__(self, json_path):
		self.json_path = json_path

	def dump_json(self, src, scale_up=1e4):
		if isinstance(src['depot_xy'], torch.Tensor):
			data = {}
			for k, v in src.items():
				data[k] = v.squeeze(0).tolist()#.numpy()

		# change names of keys in dict 
		# https://note.nkmk.me/python-dict-change-key/
		data['vehicle_capacities'] = data.pop('car_capacity')
		data['starts'] = data.pop('car_start_node')
		data['ends'] = data['starts']

		# build distance matrix
		data = get_dist_mat(data, scale_up)

		# add extra keys and values
		data['num_locations'] = len(data['dist_mat'])
		data['num_vehicles'] = len(data['vehicle_capacities'])

		# data['demand'] --> data['demands'] which includes depot location
		data['demands'] = [0] * len(data['dist_mat'])# https://note.nkmk.me/python-list-initialize/
		cnt = 0
		for i in range(data['num_locations']):	
			if i in set(data['starts']):
				cnt += 1
			else:
				data['demands'][i] = data['demand'][i-cnt]
		del data['demand']
		assert len(data['demands']) == data['num_locations'], 'should be same'
		
		# scale up demand and capacity by 10^4
		data['vehicle_capacities'] = list(map(lambda x: int(scale_up*x), data['vehicle_capacities']))
		data['demands'] = list(map(lambda x: int(scale_up*x), data['demands']))

		with open(self.json_path, 'w') as f:
			json.dump(data, f, indent = 4)

	def load_json(self):
		if not os.path.isfile(self.json_path):
			raise FileNotFoundError
		with open(self.json_path) as f:
			data = json.load(f)
			return data
		for k, v in dst.items():
			dst[k] = v.tolist()
		return dst
		

def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
	# return math.sqrt((x2-x1)**2+(y2-y1)**2)# pow(hoge,2)
	
def get_dist_mat(data, scale_up = 1e4):# digit = 2 
	xy = np.concatenate([data['depot_xy'], data['customer_xy']], axis = 0)
	n = len(xy)
	dist = [[0. for i in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(i, n):
			arc = get_dist(xy[i], xy[j])
			dist[i][j] = dist[j][i] = int(arc*scale_up)#round(float(two), digit)
	# meaning that we will have 4 decimal digits of precision

	data['dist_mat'] = dist
	del data['depot_xy']
	del data['customer_xy']
	return data

if __name__ == '__main__':
	device = torch.device('cpu')# torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
	n_depot = 2
	n_car_each_depot = 6#12
	n_customer = 50
	seed = 123
	capa = 1.

	from Torch.dataset import generate_data
	data = generate_data(device, batch = 1, n_car_each_depot = n_car_each_depot, n_depot = n_depot, n_customer = n_customer, capa = capa, seed = seed)
	
	basename = f'n{n_customer}d{n_depot}c{n_car_each_depot}D{int(capa)}.json'
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