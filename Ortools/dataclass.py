import os
import numpy as np
import torch
import math
import json
import sys

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
		# return math.sqrt((x2-x1)**2+(y2-y1)**2)# pow(hoge,2)
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
	
	
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

