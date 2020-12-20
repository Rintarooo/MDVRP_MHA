import numpy as np
import math
import json

CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50.}

def generate_data(n_car = 15, n_depot = 1, n_customer = 20, seed = None):
	if seed is not None:
		np.random.seed(seed = seed)
	n_node = n_depot + n_customer
	assert (1. * n_car >= 9. / CAPACITIES[n_customer]) * n_customer, 'infeasible; Vechile Capacity should be larger than Customer Demand' 
	return {'depot_xy': np.random.rand(n_depot, 2)
			,'customer_xy': np.random.rand(n_customer, 2)
			,'demand': np.random.randint(1, 10, n_customer) / CAPACITIES[n_customer]# 1~10 discrete digit
			,'car_start_node': np.random.randint(0, n_depot, n_car)
			,'car_capacity': np.ones(n_car)
			}

def dump_json(data, json_path):
	"""
	https://note.nkmk.me/python-json-load-dump/
	"""
	if isinstance(data['depot_xy'], np.ndarray):
		dst = {}
		for k, v in data.items():
			dst[k] = v.tolist()# since torch tensor can't be converted into json
	with open(json_path, 'w') as f:
		json.dump(dst, f, indent = 4)

def dump_custom_json(json_path, new_json_path, scale_up = 1e4):
	
"""
def load_data(path: str) -> dict:
	import yaml
	with open(path) as file:
		data = yaml.safe_load(file)
	data['num_locations'] = len(data['demands'])
	data['num_vehicles'] = len(data['vehicle_capacities'])
	return data
"""




		# with open(path, 'r') as f:
		# 	# lines = list(map(lambda s: s.strip(), f.readlines()))
		# 	lines = f.readlines()
		# 	depot_xy, customer_xy, demand, car_capacity, car_start_node = [[] for _ in range(5)]
			
		# 	for i, line in enumerate(lines):
		# 		line = line.strip()

		# 		if (i == 0):
		# 			print('line0: ', line)
		# 			n_car_each_depot, n_customer, n_depot = list(map(lambda z: int(z), line.split()))

		# 		elif (1 <= i and i <= n_depot):
		# 			print('line1: ', line)
		# 			max_duration, max_load = list(map(lambda k: float(k)/100., line.split()))
		# 			car_capacity.append(max_load)

		# 		elif (n_depot+1 <= i and i <= n_customer+n_depot):
		# 			print('line2: ', line)
		# 			arr = line.split()
		# 			_, x, y, service_duration, dema = list(map(lambda k: float(k)/100., arr[:5]))
		# 			tmp = []
		# 			tmp.append(x)
		# 			tmp.append(y)
		# 			customer_xy.append(tmp)
		# 			demand.append(dema)
				
		# 		elif (n_depot+n_customer+1 <= i and i <= n_depot+n_customer+n_depot):
		# 			print('line3: ', line)
		# 			arr = line.split()
		# 			_, x, y = list(map(lambda k: float(k)/100., arr[:3]))
		# 			tmp = []
		# 			tmp.append(x)
		# 			tmp.append(y)
		# 			depot_xy.append(tmp)

		# # print(np.array(depot_xy).shape)
		# # print(np.array(customer_xy).shape)
		# # print(np.array(demand).shape)
		
		# for i in range(n_depot):
		# 	for j in range(n_car_each_depot):
		# 		car_start_node.append(i)

		# # make the list n_depot times longer
		# car_capacity = car_capacity * n_car_each_depot
		
		# assert len(car_capacity) == n_depot * n_car_each_depot, 'n_car'
		# assert len(car_capacity) == len(car_start_node), 'n_car'
		# assert len(demand) == n_customer, 'n_customer'

		
		# data = {}
		# data['depot_xy'] = torch.tensor(np.array(depot_xy), dtype = torch.float)
		# data['customer_xy'] = torch.tensor(np.array(customer_xy), dtype = torch.float) 
		# data['demand'] = torch.tensor(np.array(demand), dtype = torch.float)
		# data['car_start_node'] = torch.tensor(np.array(car_start_node), dtype = torch.long)
		# data['car_capacity'] = torch.tensor(np.array(car_capacity), dtype = torch.float)	
		
		# if return_torch:
		# 	dst = {}
		# 	for k, v in data.items():
		# 		dst[k] = v.unsqueeze(0)# unsqueeze towards batch size
		# 		print(k, v.unsqueeze(0).size())
		# 	return dst

		# elif json_path is not None:
		# 	dump_json(data, json_path)
		# 	# dump_custom_json(json_path, new_json_path)
		# 	# data = load_json(new_json_path)

if __name__ == '__main__':
	n_car, n_depot, n_customer, seed = 15, 4, 20, 123
	data = generate_data(n_car, n_depot, n_customer, seed)
	
	json_path = 'samp.json'
	new_json_path = 'samp_new.json'
	dump_json(data, json_path)
	dump_custom_json(json_path, new_json_path)
	data = load_json(new_json_path)
	print(data)
