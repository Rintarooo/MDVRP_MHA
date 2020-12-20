#ifndef cluster_hpp
#define cluster_hpp
#include "customer.hpp"
#include "depot.hpp"
#include <vector>
#include <map>

float createClusters(std::vector<Depot>& depots, std::vector<Customer>& customers, spChromosome& initialPopulation, Swappable& swappable, int scale[]);

#endif /* cluster_hpp */
