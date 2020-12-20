from ortools.constraint_solver.pywrapcp \
	import Assignment, DefaultRoutingSearchParameters,\
	RoutingDimension, RoutingIndexManager, RoutingModel
from ortools.constraint_solver.routing_enums_pb2 \
	import FirstSolutionStrategy, LocalSearchMetaheuristic

def node_properties(
	data: dict,
	manager: RoutingIndexManager,
	assignment: Assignment,
	capacity_dimension: RoutingDimension,
	time_dimension: RoutingDimension,
	index: int,
) -> tuple:
	"""
	Get a node's properties corresponding to the index.
	"""
	node_index = manager.IndexToNode(index)
	demand = data['demands'][node_index]
	load = assignment.Value(capacity_dimension.CumulVar(index))
	if 'time_windows' in data.keys():
		time_var = time_dimension.CumulVar(index)
		time = assignment.Value(time_dimension.CumulVar(index))
		time_min, time_max = assignment.Min(time_var), assignment.Max(time_var)
		return (node_index, demand, time_min, time_max, load, time)
	return (node_index, demand, load)

def print_solution(
	data: dict,
	routing: RoutingModel,
	manager: RoutingIndexManager,
	assignment: Assignment,
) -> None:
	capacity_dimension = routing.GetDimensionOrDie('Capacity')
	time_dimension = None
	total_cost = 0
	if 'time_windows' in data.keys():
		time_dimension = routing.GetDimensionOrDie('Time')
		total_time = 0

	for vehicle_id in range(data['num_vehicles']):
		index = routing.Start(vehicle_id)
		node_props = []

		while not routing.IsEnd(index):
			props = node_properties(data, manager, assignment, capacity_dimension, time_dimension, index)
			node_props.append(props)
			index = assignment.Value(routing.NextVar(index))

		props = node_properties(data, manager, assignment, capacity_dimension, time_dimension, index)
		node_props.append(props)

		# arc weight
		cost = 0 
		node_indexes = [prop[0] for prop in node_props]
		for i in range(len(node_indexes)-1):
			idx_from = node_indexes[i]
			idx_to = node_indexes[i+1]
			cost += data['dist_mat'][idx_from][idx_to]
		total_cost += cost

		if 'time_windows' in data.keys():
			route_time = assignment.Value(time_dimension.CumulVar(index))
			route = "\n  -> ".join(['[Node %2s(%s)TW(%2s, %2s): Vehicle Load(%2s) and Arrived(%2s)]' % prop for prop in node_props])
			plan_output = (f'Route for vehicle {vehicle_id}:\n     {route}\n'
						   f'Cost: {cost}\nLoad: {props[-1]}\nTime: {route_time} min\n')
			print(plan_output)

			total_time += route_time
		else:
			route = "\n  -> ".join(['[Node %2s(%s): Vehicle Load(%2s)]' % prop for prop in node_props])
			plan_output = (f'Route for vehicle {vehicle_id}:\n     {route}\n'
						   f'Cost: {cost}\nLoad: {props[-1]}\n')
			print(plan_output)

	print(f'Total cost of all routes: {total_cost}\n')
	if 'time_windows' in data.keys():
		print('*** format ***: \n[Node node_index(demand)TW(Time Window min, max): Vehicle Load(accumulated load) and Arrived(time)]')
		print(f'Total time of all routes: {total_time} min')
