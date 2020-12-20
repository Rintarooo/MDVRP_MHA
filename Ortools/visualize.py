from ortools.constraint_solver.pywrapcp \
	import Assignment, DefaultRoutingSearchParameters,\
	RoutingDimension, RoutingIndexManager, RoutingModel
from ortools.constraint_solver.routing_enums_pb2 \
	import FirstSolutionStrategy, LocalSearchMetaheuristic

import pygraphviz as pgv

def draw_network_graph(data: dict, filename: str = 'network.png', prog: str = 'dot') -> None:
	"""
	Draw a network graph of the problem.
	"""
	dist_mat = data['dist_mat']
	demands = data['demands']
	if 'time_windows' in data.keys():
		time_windows = data['time_windows']
	n_loc = data['num_locations']
	graph = pgv.AGraph(directed=False)

	def _node(index: int) -> str:
		if index == 0:
			return f'{index}\nDepot'
		if 'time_windows' in data.keys():
			return f'{index}\nDemand: {demands[index]}\nRange: {time_windows[index]}'
		return f'{index}\nDemand: {demands[index]}'

	for i in range(n_loc):
		for j in range(i + 1, n_loc):
			weight = dist_mat[i][j]
			graph.add_edge(_node(i), _node(j), weight=weight, label=weight)

	graph.draw(filename, prog=prog)
	print(f'The network graph has been saved to {filename}.')

def draw_route_graph(
	data: dict,
	routing: RoutingModel,
	manager: RoutingIndexManager,
	assignment: Assignment,
	filename: str = 'route.png',
	prog='sfdp',
) -> None:
	"""
	Draw a route graph based on the solution of the problem.
	"""

	dist_mat = data['dist_mat']
	demands = data['demands']
	if 'time_windows' in data.keys():
		time_windows = data['time_windows']
	graph = pgv.AGraph(directed=True)

	def _node(index: int) -> str:
		if index == 0:
			return f'{index}\nDepot'
		if 'time_windows' in data.keys():
			return f'{index}\nDemand: {demands[index]}\nRange: {time_windows[index]}'
		return f'{index}\nDemand: {demands[index]}'

	for vehicle_id in range(data['num_vehicles']):
		index = routing.Start(vehicle_id)
		while not routing.IsEnd(index):
			node_index = manager.IndexToNode(index)
			next_index = assignment.Value(routing.NextVar(index))
			next_node_index = manager.IndexToNode(next_index)
			weight = dist_mat[node_index][next_node_index]
			graph.add_edge(_node(node_index), _node(next_node_index), weight=weight, label=weight)
			index = next_index

	graph.draw(filename, prog=prog)
	print(f'The route graph has been saved to {filename}.')
