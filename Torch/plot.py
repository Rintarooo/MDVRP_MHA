from time import time
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from dataset import generate_data
from baseline import load_model
from config import test_parser

import sys
sys.path.append('../')
from dataclass import TorchJson

# def get_clean_path(arr):
# 	"""Returns extra zeros from path.
# 	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
# 	"""
# 	p1, p2 = 0, 1
# 	output = []
# 	while p2 < len(arr):
# 		if arr[p1] != arr[p2]:
# 			output.append(arr[p1])
# 			if p2 == len(arr) - 1:
# 				output.append(arr[p2])
# 		p1 += 1
# 		p2 += 1

# 	if output[0] != 0:
# 		output.insert(0, 0)# insert 0 in 0th of the array
# 	if output[-1] != 0:
# 		output.append(0)# insert 0 at the end of the array
# 	return output

def clear_route(arr):
	dst = []
	for i in range(len(arr)-1):
		if arr[i] != arr[i+1]:
			dst.append(arr[i])
	if len(dst) > 0:
		dst.append(dst[0])
	return dst

def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
	
def get_dist_mat(xy): 
	n = len(xy)
	dist_mat = [[0. for i in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(i, n):
			dist = get_dist(xy[i], xy[j])
			dist_mat[i][j] = dist_mat[j][i] = dist#round(float(two), digit)
	return dist_mat

def opt2_swap(route, dist_mat): 
	size = len(route)
	improved = True
	while improved:
		improved = False
		for i in range(size - 2):
			i1 = i + 1
			a = route[i]
			b = route[i1]
			for j in range(i + 2, size):
				j1 = j + 1
				if j == size - 1:
					j1 = 0

				c = route[j]
				d = route[j1]
				if i == 0 and j1 == 0: continue# if i == j1
				if(dist_mat[a][c] + dist_mat[b][d] < dist_mat[a][b] + dist_mat[c][d]):
					""" i i+1 j j+1
						swap(i+1, j)
					"""
					tmp = route[i1:j1]
					route[i1:j1] = tmp[::-1]# tmp in inverse order
					improved = True 
	return route

def apply_2opt(tup):
	routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch = tup
	dist_mat = get_dist_mat(xy)
	new_routes = []
	for route in routes:# apply 2-opt to each route
		if len(route) > 0: new_routes.append(opt2_swap(route, dist_mat))
	print('routes(2opt): ', new_routes)
	
	cost = 0.
	for i, route in enumerate(new_routes, 1):
		coords = xy[[int(x) for x in route]]
		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		total_length = np.sum(lengths)
		cost += total_length
	return (new_routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch)
	

def get_more_info(cost, pi, idx_in_batch):
	
	# Remove unneeded values
	routes = []
	for pi_of_each_car in pi:
		route = clear_route(pi_of_each_car)
		if len(route) > 0:
			routes.append(route)
	print('routes: ', routes)
	
	# data.keys(), ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']
	depot_xy = data['depot_xy'][idx_in_batch].cpu().numpy()
	customer_xy = data['customer_xy'][idx_in_batch].cpu().numpy()
	demands = data['demand'][idx_in_batch].cpu().numpy()
	xy = np.concatenate([depot_xy, customer_xy], axis = 0)
	return (routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch)

def plot_route(tup, title):
	routes, depot_xy, customer_xy, demands, xy, cost, pi, idx_in_batch = tup

	customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
	path_traces = []
	for i, route in enumerate(routes, 1):
		coords = xy[[int(x) for x in route]]

		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		total_length = np.sum(lengths)
		
		path_traces.append(go.Scatter(x = coords[:, 0],
									y = coords[:, 1],
									mode = 'markers+lines',
									name = f'Vehicle{i}: Length = {total_length:.3f}',
									opacity = 1.0))
	
	trace_points = go.Scatter(x = customer_xy[:, 0],
							  y = customer_xy[:, 1],
							  mode = 'markers+text', 
							  name = 'Customer (demand)',
							  text = customer_labels,
							  textposition = 'top center',
							  marker = dict(size = 7),
							  opacity = 1.0
							  )

	trace_depo = go.Scatter(x = depot_xy[:,0],
							y = depot_xy[:,1],
							# mode = 'markers+text',
							mode = 'markers',
							# name = 'Depot (Capacity = 1.0)',
							name = 'Depot',
							# text = ['1.0'],
							# textposition = 'bottom center',
							marker = dict(size = 23),
							marker_symbol = 'triangle-up'
							)
	
	layout = go.Layout(
						# title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
						title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', xref = 'paper', yref = 'paper', pad = dict(b = 10)),
						# xaxis = dict(title = 'X', range = [0, 1], ticks='outside'),
						# yaxis = dict(title = 'Y', range = [0, 1], ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
						xaxis = dict(title = 'X', range = [0, 1], linecolor = 'black', showgrid=False, ticks='outside', linewidth=1, mirror=True),
						yaxis = dict(title = 'Y', range = [0, 1], linecolor = 'black', showgrid=False, ticks='outside', linewidth=1, mirror=True),
						showlegend = True,
						width = 750,
						height = 700,
						autosize = True,
						template = "plotly_white",
						legend = dict(x = 1.05, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = 'black', borderwidth = 1)
						# legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						# legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
						)

	data = [trace_points, trace_depo] + path_traces
	fig = go.Figure(data = data, layout = layout)
	fig.show()

if __name__ == '__main__':
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pretrained = load_model(device, args.path, embed_dim = 128, n_encode_layers = 3)
	print(f'model loading time:{time()-t1}s')
	
	t2 = time()
	if args.txt is not None:
		hoge = TorchJson(args.txt)
		data = hoge.load_json(device)# return tensor on GPU
		for k, v in data.items():
			shape = (args.batch, ) + v.size()[1:] 
			data[k] = v.expand(*shape).clone()
			# print('k, v', k, *v.size())
			# print(*shape)
		
	else:
		data = {}
		for k in ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']:
			elem = [generate_data(device, batch = 1, n_car = args.n_car, n_depot = args.n_depot, n_customer = args.n_customer, seed = args.seed)[k].squeeze(0) for j in range(args.batch)]
			data[k] = torch.stack(elem, 0)
	
	# for k, v in data.items():
	# 	print('k, v', k, v.size())
	# 	print(v.type())# dtype of tensor
	
	print(f'data generate time:{time()-t1}s')
	pretrained = pretrained.to(device)
	# data = list(map(lambda x: x.to(device), data))
	pretrained.eval()
	with torch.no_grad():
		costs, _, pis = pretrained(data, return_pi = True, decode_type = args.decode_type)
	# print('costs:', costs)
	idx_in_batch = torch.argmin(costs, dim = 0)
	cost = costs[idx_in_batch].cpu().numpy()
	if args.write_csv is not None:
		with open(args.write_csv, 'a') as f:
			f.write(f'{time()-t1},{time()-t2},{cost}\n')
	print(f'decode type: {args.decode_type}\nminimum cost(without 2opt): {cost:.3f}\nidx: {idx_in_batch} out of {args.batch} solutions')
	print(f'\ninference time: {time()-t1}s')
	print(f'inference time(without loading model): {time()-t2}s')
	
	pi = pis[idx_in_batch].cpu().numpy()
	tup = get_more_info(cost, pi, idx_in_batch)
	if args.write_csv is None:
		title = 'Pretrained'
		plot_route(tup, title)
		print('plot time: ', time()-t1)
		print(f'plot time(without loading model): {time()-t2}s')

	tup = apply_2opt(tup)
	cost = tup[-3]
	if args.write_csv_2opt is not None:
		with open(args.write_csv_2opt, 'a') as f:
			f.write(f'{time()-t1},{time()-t2},{cost}\n')
	print(f'minimum cost(without 2opt): {cost:.3f}')
	print('inference time: ', time()-t1)
	print(f'inference time(without loading model): {time()-t2}s')

	if args.write_csv_2opt is None:
		title = 'Pretrained'
		plot_route(tup, title)
		print('plot time: ', time()-t1)
		print(f'plot time(without loading model): {time()-t2}s')