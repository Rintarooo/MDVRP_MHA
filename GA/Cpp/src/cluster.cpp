#include "cluster.hpp"
#include <iostream>
#include <cmath>

bool pairCompare( std::pair<int,float> i, std::pair<int,float> j)
{
    return i.second < j.second;
}


float createClusters(std::vector<Depot>& depots, std::vector<Customer>& customers, spChromosome& initialPopulation, Swappable& swappable, int scale[])
{
    int minX = scale[0];
    int maxX = scale[2];
    int minY = scale[1];
    int maxY = scale[3];
    float pyta = sqrt(pow((maxX-minX),2) + pow((maxY-minY),2));
    
    for(std::vector<Depot>::iterator it = depots.begin(); it != depots.end(); ++it) {
        (*initialPopulation)[(*it).depotNumber] = std::vector<Customer*>();
    }
    
    for(std::vector<Customer>::iterator it = customers.begin(); it != customers.end(); ++it) {
        std::map<int, float> distances;
        for(std::vector<Depot>::iterator it2 = depots.begin(); it2 != depots.end(); ++it2) {
            distances[(*it2).depotNumber] = sqrt(pow((*it).x - (*it2).getX(), 2.0) + pow((*it).y - (*it2).getY(), 2.0));
        }
        
        std::pair<int,float> min = *min_element(distances.begin(), distances.end(), pairCompare );
        (*initialPopulation)[min.first].push_back(&(*it));
        
        swappable[(*it).customerNumber] = std::vector<Depot>();
        swappable[(*it).customerNumber].push_back(depots[min.first-1]);
        
        for(std::map<int, float>::iterator distanceIt = distances.begin(); distanceIt != distances.end(); distanceIt++)
        {
            if (distanceIt->first != min.first)
            {
                if (distanceIt->second/min.second <= 3.0 || distanceIt->second/pyta < 0.1)
                {
                    swappable[(*it).customerNumber].push_back(depots[distanceIt->first-1]);
                }
            }
        }
        
        if(swappable[(*it).customerNumber].size() <= 1)
        {
            swappable.erase((*it).customerNumber);
        }
        
    }
    return pyta;
    
}


