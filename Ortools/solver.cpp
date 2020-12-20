// https://developers.google.com/optimization/routing/vrp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using ll = long long;
using namespace std;

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"
using namespace operations_research;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

json load_json(const char* filename){
	ifstream ifs(filename);
	if(!ifs) exit(1);
	json data = json::parse(ifs);
	// https://developers.google.com/optimization/routing/routing_tasks#c++_5
	// data["vehicle_capacities"] = data["car_capacity"];
	// data["starts"] = data["car_start_node"];
	// data["ends"] = data["starts"];
	// data["demands"] = data["demand"];
	
	// // # build distance matrix
	// // data = get_dist_mat(data, scale_up)
	// // # scale up demand and capacity by 10^4
	// // data['vehicle_capacities'] = list(map(lambda x: int(scale_up*x), data['vehicle_capacities']))
	// // data['demands'] = list(map(lambda x: int(scale_up*x), data['demands']))
	
	// // add extra keys and values
	// data["num_locations"] = data["demand"].size();
	// data["num_vehicles"] = data["vehicle_capacities"].size();
	return data;
}

int main(){
// int main(argc, char *argv[]){
	// Instantiate the data problem
	// load data from json file
	const char *filename;
	filename = "samp_new.json";// = argv[1];
	json data = load_json(filename);
	
	// Create Routing Index Manager
	RoutingIndexManager manager(data["num_locations"], data["num_vehicles"], data["starts"]);//, data["ends"]);

	// // Create Routing Model.
	// RoutingModel routing(manager);

	return 0;
}