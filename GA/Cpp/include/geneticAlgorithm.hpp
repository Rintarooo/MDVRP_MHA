#ifndef geneticAlgorithm_hpp
#define geneticAlgorithm_hpp
#include <vector>
#include "customer.hpp"
#include "depot.hpp"
#include <map>
#include "route.hpp"
// #include "Graphics.hpp"

#include "types.h"

using Population = std::vector<std::tuple<spChromosome, float>>;
// Population population
// std::vector<std::tuple<chrom, fitness>>;
// spChromosome chrom = std::get<0>(population[0]);
// "spChromosome" defined in customer.hpp
// float fitness = std::get<1>(population[0])

class GeneticAlgorithm
{
    int *scale;
    float mapSize;
    float crossoverRate, intraMutationRate, interMutationRate, elitismRate;
    int interMutationInterval;
    int penalizeRoutesUntil = 0;
    
    std::vector<Depot>& depots;
    std::vector<Customer>& customers;
    int preferredVehicleLimit;
    // Graphics* graphics;
    
    //float cost(
    upRouteMap deconstructRoutes(spChromosome& chromosome);
    spChromosome reconstructChromosome(upRouteMap& routes);
    float cost(upRouteMap& routes);
    // float cost2(upRouteMap& routes);
    float fitness(upRouteMap& routes, int gen);
    
    float euclidDistance(float x1, float y1, float x2, float y2);
    int getRandomChromoKey(spChromosome& chromo);
    spChromosome mutation(spChromosome& oldChromo);
    spChromosome interMutation(spChromosome& oldChromo);
    
    spChromosome insertRouteToDepot(spChromosome& oldChromo, int from, int to);
    spChromosome swapMutation(spChromosome& oldChromo);
    spChromosome reverseMutation(spChromosome& oldChromo);
    spChromosome rerouteMutation(spChromosome& oldChromo);
    spChromosome depotSwapMutation(spChromosome& oldChromo);
    spChromosome depotInsertMutation(spChromosome& oldChromo);
    spChromosome swapBestMutation(spChromosome& oldChromo);
    void generateRandomPopulation(Population& population, int populationSize);
    spChromosome select(Population& population);
    void applyElitism(Population& oldPop, Population& newPop);
    std::tuple<spChromosome, spChromosome> crossover(spChromosome& p1, spChromosome& p2);
    spChromosome insertBestFeasable(spChromosome oldChromo, int, Customer*);
    spChromosome initialChromosome;
    Swappable swappable;
    
public:
    GeneticAlgorithm(std::vector<Depot>& depots, std::vector<Customer>& customers, int scale[], int preferredVehicleLimit);
    ~GeneticAlgorithm()
    {
        // delete graphics;
    }
    
    void run(int generationsStepOne, int generationsStepTwo, int populationSize);
};

#endif /* geneticAlgorithm_hpp */