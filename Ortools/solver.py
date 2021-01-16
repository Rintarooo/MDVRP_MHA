# shell --> python solver.py -p samp.json
import argparse
from typing import Callable
from time import time
from ortools.constraint_solver.pywrapcp \
	import Assignment, DefaultRoutingSearchParameters,\
	RoutingDimension, RoutingIndexManager, RoutingModel
from ortools.constraint_solver.routing_enums_pb2 \
	import FirstSolutionStrategy, LocalSearchMetaheuristic

# import sys
# sys.path.append('../')
from dataclass import OrtoolsJson# from data import load_json

from print import print_solution
# from visualize import draw_network_graph, draw_route_graph

TransitCallback = Callable[[int, int], int]
UnaryTransitCallback = Callable[[int], int]

def create_distance_callback(manager: RoutingIndexManager, data: dict) -> TransitCallback:
	"""
	Create a callback to return the weight between nodes.
	"""
	def distance_callback(from_index: int, to_index: int) -> int:
		"""
		Return the weight between the two nodes.
		"""
		from_node = manager.IndexToNode(from_index)
		to_node = manager.IndexToNode(to_index)
		return data['dist_mat'][from_node][to_node]
	return distance_callback

def create_demand_callback(manager: RoutingIndexManager, data: dict) -> UnaryTransitCallback:
	"""
	Create a callback to get demands at each location.
	"""
	def demand_callback(from_index: int) -> int:
		"""
		Return the demand.
		"""
		from_node = manager.IndexToNode(from_index)
		return data['demands'][from_node]
	return demand_callback

def add_capacity_constraints(
	routing: RoutingModel,
	manager: RoutingIndexManager,
	data: dict,
	demand_callback_index: int,
) -> None:
	"""
	Since the capacity constraints involve 
	the weight of the load a vehicle is carrying
	a quantity that accumulates over the route
	we need to create a dimension for capacities
	"""
	routing.AddDimensionWithVehicleCapacity(
		demand_callback_index,
		slack_max=0,  # null capacity slack
		vehicle_capacities=data['vehicle_capacities'],  # vehicle maximum capacities
		fix_start_cumul_to_zero=True,  # start cumul to zero
		name='Capacity',)

def create_time_callback(manager: RoutingIndexManager, data: dict) -> TransitCallback:
	"""
	Create a callback to get total times between locations.
	"""
	def time_callback(from_index: int, to_index: int) -> int:
		"""
		Return the total time between the two nodes.
		"""
		from_node = manager.IndexToNode(from_index)
		to_node = manager.IndexToNode(to_index)
		# Get the service time to the specified location
		serv_time = data['service_times'][from_node]
		# Get the travel times between two locations
		trav_time = data['dist_mat'][from_node][to_node]
		return serv_time + trav_time
	return time_callback

def add_time_window_constraints(
	routing: RoutingModel,
	manager: RoutingIndexManager,
	data: dict,
	time_callback_index: int,
) -> None:
	max_ = 120
	routing.AddDimension(
		time_callback_index,
		slack_max=max_,  # An upper bound for slack (the wait times at the locations)
		capacity=max_,  # An upper bound for the total time over each vehicle's route
		# Don't force start cumul to zero. This doesn't have any effect in this example,
		# since the depot has a start window of (0, 0).
		fix_start_cumul_to_zero=False,
		name='Time',
	)
	time_dimension = routing.GetDimensionOrDie('Time')
	for loc_idx, (open_time, close_time) in enumerate(data['time_windows']):
		index = manager.NodeToIndex(loc_idx)
		time_dimension.CumulVar(index).SetRange(open_time, close_time)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', help='JSON file path of data')
	parser.add_argument('-g', '--graph', help='export images of the network and the routes of vehicles', action='store_true')
	parser.add_argument('--gls',
		help="enable Guided Local Search (Note: This could take a long time, so it's a good idea "
			  'to use --gls with -v to see the progress of a search)',
		action='store_true')
	parser.add_argument('-v', '--verbose', help='enable verbose output', action='store_true')
	parser.add_argument('-c', '--write_csv', help='export csv')
	return parser.parse_args()

if __name__ == '__main__':
	# Parse command line arguments
	args = parse_args()
	t1 = time()
	# Instantiate the data problem
	# data = load_data(args.path)
	if args.path is not None:
		hoge = OrtoolsJson(args.path)
		data = hoge.load_json()
	else:
		raise FileNotFoundError(f'json file: {args.path}')
	# Create Routing Index Manager
	manager = RoutingIndexManager(data['num_locations'], data['num_vehicles'], 
									data['starts'], data['ends'])
	# Create Routing Model
	routing = RoutingModel(manager)

	"""
	Define weight of each edge
	https://developers.google.com/optimization/routing/vrp
	"""
	distance_callback = create_distance_callback(manager, data)
	transit_callback_index = routing.RegisterTransitCallback(distance_callback)
	routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

	"""
	Add capacity constraints
	https://developers.google.com/optimization/routing/cvrp
	"""
	demand_callback = create_demand_callback(manager, data)
	demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
	add_capacity_constraints(routing, manager, data, demand_callback_index)

	"""
	Add time window constraints
	https://developers.google.com/optimization/routing/vrptw
	"""
	if 'time_windows' in data.keys():
		time_callback = create_time_callback(manager, data)
		time_callback_index = routing.RegisterTransitCallback(time_callback)
		add_time_window_constraints(routing, manager, data, time_callback_index)

	"""
	The code sets the first solution strategy to PATH_CHEAPEST_ARC, 
	which creates an initial route for the solver by repeatedly adding edges with the least weight 
	that don't lead to a previously visited node (other than the depot).
	For other options, see the link below
	https://developers.google.com/optimization/routing/routing_options#first_sol_options
	"""
	search_params = DefaultRoutingSearchParameters()
	search_params.first_solution_strategy = FirstSolutionStrategy.PATH_CHEAPEST_ARC
	
	if args.gls:
		search_params.local_search_metaheuristic = LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
		# NOTE: Since Guided Local Search could take a very long time, we set a reasonable time limit
		search_params.time_limit.seconds = 30
	
	if args.verbose:
		search_params.log_search = True

	# Solve the problem
	assignment = routing.SolveWithParameters(search_params)
	
	if routing.status() == 1 and assignment:
		t2 = time()
		# Print the solution
		best_score = None
		best_score = print_solution(data, routing, manager, assignment)
		print(f'Measured Execute time: {round(t2-t1, 6)}sec')
		if args.write_csv is not None:
			with open(args.write_csv, 'a') as f:
				f.write(f'{t2-t1},{best_score/1e4}\n')

		# Draw network and route graphs
		if args.graph:
			from visualize import draw_network_graph, draw_route_graph
			# draw_network_graph(data)
			draw_route_graph(data, routing, manager, assignment)
	else:
		print('routing.status(): ', routing.status())
		print('refer to the link below in regards to search status')
		print('https://developers.google.com/optimization/routing/routing_options')
			
