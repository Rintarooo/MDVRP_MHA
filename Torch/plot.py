from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from dataset import generate_data
from baseline import load_model
from config import test_parser

import sys
sys.path.append('../')
from dataclass import TorchJson


# python plot.py -b 128 -dt sampling -p Weights/VRP50_train_epoch75.pt -t data/

def get_clean_path(arr):
	"""Returns extra zeros from path.
	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
	"""
	p1, p2 = 0, 1
	output = []
	while p2 < len(arr):
		if arr[p1] != arr[p2]:
			output.append(arr[p1])
			if p2 == len(arr) - 1:
				output.append(arr[p2])
		p1 += 1
		p2 += 1

	if output[0] != 0:
		output.insert(0, 0)# insert 0 in 0th of the array
	if output[-1] != 0:
		output.append(0)# insert 0 at the end of the array
	return output

def clear_each_route(arr):
	dst = []
	for i in range(len(arr)-1):
		if arr[i] != arr[i+1]:
			dst.append(arr[i])
	if len(dst) > 0:
		dst.append(dst[0])
	return dst

def plot_route(data, pi, costs, title, idx_in_batch = 0):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""
	cost = costs[idx_in_batch].cpu().numpy()
	
	# Remove extra zeros
	list_of_paths = []
	pi_cars = pi[idx_in_batch].cpu().numpy()
	for each_car_pi in pi_cars:
		list_of_paths.append(clear_each_route(each_car_pi))
	print(list_of_paths)

	# ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']
	depot_xy = data['depot_xy'][idx_in_batch].cpu().numpy()
	customer_xy = data['customer_xy'][idx_in_batch].cpu().numpy()
	demands = data['demand'][idx_in_batch].cpu().numpy()
	# customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
	customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
	
	xy = np.concatenate([depot_xy, customer_xy], axis = 0)

	path_traces = []
	for i, path in enumerate(list_of_paths, 1):
		coords = xy[[int(x) for x in path]]

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
							name = 'Depot (Capacity = 1.0)',
							# text = ['1.0'],
							# textposition = 'bottom center',
							marker = dict(size = 23),
							marker_symbol = 'triangle-up'
							)
	
	layout = go.Layout(title = dict(text = f'<b>VRP{customer_xy.shape[0]} depot{depot_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
					   # xaxis = dict(title = 'X', range = [0, 1], ticks='outside'),
					   # yaxis = dict(title = 'Y', range = [0, 1], ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
					   xaxis = dict(title = 'X', range = [0, 1], linecolor = 'black', showgrid=False, ticks='outside', linewidth=1, mirror=True),
					   yaxis = dict(title = 'Y', range = [0, 1], linecolor = 'black', showgrid=False, ticks='outside', linewidth=1, mirror=True),
					   showlegend = True,
					   width = 750,
					   height = 700,
					   autosize = True,
					   template = "plotly_white",
					   legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
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
		costs, _, pi = pretrained(data, return_pi = True, decode_type = args.decode_type)
		print('costs:', costs)
		idx_in_batch = torch.argmin(costs, dim = 0)
		print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
		print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
		plot_route(data, pi, costs, 'Pretrained', idx_in_batch)
		