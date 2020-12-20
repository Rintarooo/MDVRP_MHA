#include "geneticAlgorithm.hpp"

#include "cluster.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "reportGenerator.hpp"

GeneticAlgorithm::GeneticAlgorithm(std::vector<Depot>& depots, std::vector<Customer>& customers, int scale[], int preferredVehicleLimit)
: depots(depots), customers(customers), scale(scale), preferredVehicleLimit(preferredVehicleLimit), initialChromosome(new Chromosome)
{
    mapSize = createClusters(depots, customers, initialChromosome, swappable, scale);
    // graphics = new Graphics(1200,900, scale, depots);
    crossoverRate = 0.20;
    intraMutationRate = 0.25;
    interMutationRate = 0.30;
    interMutationInterval = 10;
    elitismRate = 0.03;
}


void GeneticAlgorithm::run(int generationsStepOne, int generationsStepTwo, int populationSize)
{
    
    Population population;
    generateRandomPopulation(population, populationSize);
    // sort based on fitness = std::get<1>(population[0])
    std::sort(population.begin(), population.end(), [](auto& a, auto& b){ return std::get<1>(a) < std::get<1>(b);});
    penalizeRoutesUntil = 0.3 * generationsStepOne;// 0.3 * 5000
    
    // for (int gen = 0; gen < generationsStepOne + generationsStepTwo && !graphics->isFinishedEarly(); gen++)
    for (int gen = 0; gen < generationsStepOne + generationsStepTwo; gen++)
    {
        if(gen == generationsStepOne)// gen == 5000
        {
            //Enter evolution step two for some last second improvements
            crossoverRate = 0.0;
            population.erase(population.begin()+int(population.size()*0.1), population.end());
            elitismRate = elitismRate*8;
            populationSize = population.size();
        }
        
        Population nextPopulation;

        int i = 0;
        while(i < populationSize)// i < 400
        {
            
            try
            {
                // 1. Selection based on fitness
                // spChromosome selected1
                auto selected1 = select(population);
                auto selected2 = select(population);
                
                // 2. CrossOver
                float prob = rand()/float(RAND_MAX);
                if(prob < crossoverRate)
                {
                    std::tuple<spChromosome, spChromosome> crossRes = crossover(selected1, selected2);
                    selected1 = std::get<0>(crossRes);
                    selected2 = std::get<1>(crossRes);
                }
                
                // 3. Mutation
                prob = rand()/float(RAND_MAX);
                if(gen % interMutationInterval != 0)
                { // Intra mutation
                    prob = rand()/float(RAND_MAX);
                    if(prob < intraMutationRate)
                    {
                        selected1 = mutation(selected1);
                    }
                    
                    prob = rand()/float(RAND_MAX);
                    if(prob < intraMutationRate)
                    {
                        selected2 = mutation(selected2);
                    }
                }
                else
                { //Inter mutation
                    prob = rand()/float(RAND_MAX);
                    if(prob<interMutationRate)
                    {
                        selected1 = interMutation(selected1);
                    }
                    
                    prob = rand()/float(RAND_MAX);
                    if(prob<interMutationRate)
                    {
                        selected2 = interMutation(selected2);
                    }
                    
                }
                  
                auto newRoute = deconstructRoutes(selected1);
                auto newRoute2 = deconstructRoutes(selected2);
                
                // 4. NextGeneration
                nextPopulation.push_back(std::make_tuple(selected1,fitness(newRoute, gen)));
                nextPopulation.push_back(std::make_tuple(selected2,fitness(newRoute2, gen)));
                i += 2;
            } catch(const char* error)
            {
                std::cout << error <<std::endl;
            }
            
        }
        
        applyElitism(population, nextPopulation);
        population = nextPopulation;
        std::sort(population.begin(), population.end(), [](auto& a, auto& b){ return std::get<1>(a) < std::get<1>(b);});
        if(gen % 100 == 0)
        {
            // graphics->draw(std::move(deconstructRoutes(std::get<0>(population[0]))));
            spChromosome chrom = std::get<0>(population[0]);
            upRouteMap routes = deconstructRoutes(chrom);
            float routeLength = cost(routes);
            
            std::cout <<"Gen:"<<gen<<" best path length: "<<routeLength<<std::endl;
        }
        
    }
    std::cout<<"\n\n\n\n\n\n\n\n\n"<<std::endl;
    
    spChromosome chrom = std::get<0>(population[0]);
    upRouteMap routes = deconstructRoutes(chrom);
    std::cout << generateReport(routes) << std::endl;
    
    // graphics->handleWindow();
    std::cin.get();
}

// Tournament with size 2
spChromosome GeneticAlgorithm::select(Population& population)
{
    
    int index0 = rand()%population.size();
    int index1 = rand()%(population.size()-1);
    
    if(index1 >= index0)
    {
        index1++;
    }
    float prob = rand()/float(RAND_MAX);
    
    // geneticAlgorithm.hpp, cand0 <std::tuple<chrom, fitness>>
    auto cand0 = population[index0];
    auto cand1 = population[index1];
    
    if (prob < 0.8)
    {
        return std::get<1>(cand0) < std::get<1>(cand1)? std::get<0>(cand0) : std::get<0>(cand1);
    }    
    return std::get<1>(cand0) < std::get<1>(cand1)? std::get<0>(cand1) : std::get<0>(cand0);
}


void GeneticAlgorithm::applyElitism(Population& oldPop, Population& newPop)
{
    std::vector<unsigned> numberRange;
    numberRange.reserve(newPop.size());
    
    for (unsigned i = 0; i < newPop.size(); i++)
    {
        numberRange.push_back(i);
    }
    std::random_shuffle(numberRange.begin(), numberRange.end());
    int toReplace = int(newPop.size()*elitismRate);
    
    for (int i = 0; i < toReplace; i++)
    {
        newPop[numberRange[i]] = oldPop[i];
    }
}
void GeneticAlgorithm::generateRandomPopulation(Population& population, int populationSize)
{
    for(int chromoI = 0; chromoI < populationSize;)
    {
        try
        {
            spChromosome newChromo = std::make_shared<Chromosome>();
            for(auto it = initialChromosome->begin(); it != initialChromosome->end(); ++it)
            // cluster.hpp, spChromosome initialChromosome
            {
                (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
                std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
                std::random_shuffle((*newChromo)[it->first].begin() , (*newChromo)[it->first].end());
                
            }
            upRouteMap rs = deconstructRoutes(newChromo);
            population.push_back(std::make_tuple(newChromo,fitness(rs,0)));
            chromoI++;
        } catch(const char* err)
        {
            std::cout << err <<std::endl;
        }
        
    }

}


float GeneticAlgorithm::euclidDistance(float x1, float y1, float x2, float y2)
{
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}


upRouteMap GeneticAlgorithm::deconstructRoutes(spChromosome& chromosome)
{
    upRouteMap routes(new routeMap);

    for(Chromosome::iterator chromoIt = chromosome->begin(); chromoIt != chromosome->end(); ++chromoIt)
    {
        Depot *currentDepot = &this->depots[chromoIt->first-1];
        (*routes)[chromoIt->first] = std::vector<std::unique_ptr<Route>>();
        // route.hpp, using routeVec = std::vector<std::unique_ptr<Route>>;
        // route.hpp, using routeMap = std::map<int, routeVec>;
        // route.hpp, using upRouteMap = std::unique_ptr<routeMap>;
        // --> upRouteMap = std::unique_ptr<std::map<int, std::vector<std::unique_ptr<Route>>>>;
        // customer.hpp, using Chromosome = std::map <int, std::vector<Customer*>>;


        float routeLength = 0;
        int routeLoad = 0;
        int serviceDuration = 0;
        float prevLoopCost = 0;

        spVec<Customer*> currentRoute (new std::vector<Customer*>);
        int other_x = currentDepot->getX();
        int other_y = currentDepot->getY();

        for(std::vector<Customer*>::iterator custIt = chromoIt->second.begin(); custIt != chromoIt->second.end(); ++ custIt)
        {
            float loopCloseCost = this->euclidDistance((*custIt)->x, (*custIt)->y, currentDepot->getX(), currentDepot->getY());
            float addedDistance = this->euclidDistance((*custIt)->x, (*custIt)->y, other_x, other_y) + loopCloseCost - prevLoopCost;
            prevLoopCost = loopCloseCost;
            other_x = (*custIt)->x;
            other_y = (*custIt)->y;

            if((currentDepot->maxDuration > 0 && routeLength + addedDistance > currentDepot->maxDuration) || (routeLoad + (*custIt)->demand > currentDepot->maxVehicleLoad) || (currentDepot->getMaxServiceDuration() > 0 && serviceDuration + (*custIt)->serviceDuration > currentDepot->getMaxServiceDuration()) )
            {
                // In case first customer is further away from depot than allowed.
                if(currentRoute->size() == 0)
                {
                    throw "Invalid route";
                    
                }
                
                std::unique_ptr<Route> r(new Route(currentDepot, routeLength, routeLoad, serviceDuration, currentRoute));
                (*routes)[chromoIt->first].push_back(std::move(r));
                currentRoute = std::make_unique<std::vector<Customer*>> ();
                routeLength = 2* loopCloseCost;
                routeLoad = (*custIt)->demand;
                serviceDuration = (*custIt)->serviceDuration;
                
                if((currentDepot->maxDuration != 0 && routeLength > currentDepot->maxDuration) || routeLoad > currentDepot->maxVehicleLoad || (currentDepot->getMaxServiceDuration() != 0 && serviceDuration + (*custIt)->serviceDuration > currentDepot->getMaxServiceDuration()))
                {
                    throw "Creating a route with customer makes it invalid";
                }

            }
            else
            {
                routeLength += addedDistance;
                routeLoad += (*custIt)->demand;
                serviceDuration += (*custIt)->serviceDuration;
            }

            currentRoute->push_back((*custIt));
        }
        if(currentRoute->size() > 0)
        {
            std::unique_ptr<Route> r(new Route(currentDepot, routeLength, routeLoad, serviceDuration, currentRoute));
            (*routes)[chromoIt->first].push_back(std::move(r));
        }
        

    }
    
    // Phase 2
    
    for(routeMap::iterator depotIt = routes->begin(); depotIt != routes->end(); depotIt++)
    {
        bool updated = false;
        do{
            updated=false;
            
        for(int i = 0; i <depotIt->second.size() && depotIt->second.size() > 1; i++)
        {
            std::unique_ptr<Route>& fromRoute = depotIt->second[i];
            std::unique_ptr<Route>& toRoute = depotIt->second[(i+1)%depotIt->second.size()];
            Customer* considered = fromRoute->peekLast();
            std::tuple<bool, int, float> addCosts = toRoute->checkConstraints(0, considered);
            if(std::get<0>(addCosts))
            {
                std::tuple<float, int> removeGains = fromRoute->removeLastGain();
                
                if(std::get<0>(removeGains) > std::get<2>(addCosts))
                {
                    fromRoute->popCustomer(removeGains);
                    toRoute->insertCustomerAtBeginning(considered, std::make_tuple(std::get<2>(addCosts), std::get<1>(addCosts)));
                    updated = true;
                }

                
            }
        }
        if(updated)
        {
            for (auto it = depotIt->second.begin(); it != depotIt->second.end(); it++)
            {
                if((*it)->noRouteCustomers() == 0)
                {
                    routeVec replace;
                    for (auto it2 = depotIt->second.begin(); it2 != depotIt->second.end(); it2++)
                    {
                        if((*it2)->noRouteCustomers() != 0)
                        {
                            replace.push_back(std::move(*it2));
                        }
                        
                    }
                    
                    depotIt->second.clear();
                    depotIt->second.reserve(replace.size());
                    for(auto r = replace.begin(); r != replace.end(); r++)
                    {
                        depotIt->second.push_back(std::move(*r));
                    }
                    break;
                }
                
            }
            
        }
    }while(updated);}

    return routes;
}

std::tuple<spChromosome, spChromosome> GeneticAlgorithm::crossover(spChromosome& p1, spChromosome& p2)
{
    int depotNo = getRandomChromoKey(p1);
    
    upRouteMap r1 = deconstructRoutes(p1);
    upRouteMap r2 = deconstructRoutes(p2);
    
    routeVec& r1Vec = (*r1).find(depotNo)->second;
    routeVec& r2Vec = (*r2).find(depotNo)->second;
    if(r1Vec.size() == 0 || r2Vec.size() == 0)
    {
        throw "Cannot crossover empty chromosome";
    }
    std::unique_ptr<Route> routeToRemove1 = std::move(r1Vec[rand()%r1Vec.size()]);
    std::unique_ptr<Route> routeToRemove2 = std::move(r2Vec[rand()%r2Vec.size()]);
    
    std::vector<Customer*> custListR1 = routeToRemove1->getCustomerList();
    std::vector<Customer*> custListR2 = routeToRemove2->getCustomerList();
    
    // c = chromosome
    spChromosome c1 = std::make_shared<Chromosome>();
    spChromosome c2 = std::make_shared<Chromosome>();
    
    
    for(auto it = p1->begin(); it != p1->end(); ++it)
    {
        (*c1)[it->first] = std::vector<Customer*>(it->second.size());
        std::copy(it->second.begin(), it->second.end(), (*c1)[it->first].begin());
        
    }
    for(auto it = p2->begin(); it != p2->end(); ++it)
    {
        (*c2)[it->first] = std::vector<Customer*>(it->second.size());
        std::copy(it->second.begin(), it->second.end(), (*c2)[it->first].begin());
        
    }
    
    
    
    for(Chromosome::iterator it = c1->begin(); it != c1->end(); it++)
    {
        it->second.erase( remove_if( begin(it->second),end(it->second),
        [&](auto x){return find(begin(custListR2),end(custListR2),x)!=end(custListR2);}), end(it->second) );
    }
    
    for(Chromosome::iterator it = c2->begin(); it != c2->end(); it++)
    {
        it->second.erase( remove_if( begin(it->second),end(it->second),
        [&](auto x){return find(begin(custListR1),end(custListR1),x)!=end(custListR1);}), end(it->second) );
    }
    
    
    
    for(std::vector<Customer*>::iterator c = custListR1.begin(); c != custListR1.end(); c++)
    {
        float prob = rand()/float(RAND_MAX);
        
        upRouteMap cRoutes2 = deconstructRoutes(c2);
        int chromoIndex = 0;
        auto& routeIt = cRoutes2->find(depotNo)->second;
        
        std::vector<std::tuple<bool, int, float, int>> feasableList;
        /*
         Todo reserve
         */
        for(routeVec::iterator rVecIt = routeIt.begin(); rVecIt != routeIt.end(); rVecIt++)
        {
            std::vector<std::tuple<bool, int, float, int>> &&res = (*rVecIt)->testInsertions(*c, &chromoIndex);
            feasableList.insert(feasableList.end(), res.begin(), res.end());
            
        }
        
        std::sort(feasableList.begin(), feasableList.end(), [](auto& a, auto& b){ return std::get<2>(a) < std::get<2>(b);});
        
        int insertionIndex = -1;
        if(prob <= 0.8)
        {
            for(auto feasableIt = feasableList.begin(); feasableIt != feasableList.end(); feasableIt++)
            {
                if(std::get<0>(*feasableIt) == true)
                {
                    insertionIndex = std::get<3>(*feasableIt);
                    break;
                }
            }
        }
        else
        {
            if(feasableList.size() == 0)
            {
                insertionIndex = 0;
            }
            else
            {
                insertionIndex = std::get<3>(*feasableList.begin());
            }
        }
        
        if(insertionIndex >= 0)
        {
            c2->find(depotNo)->second.insert(c2->find(depotNo)->second.begin() + insertionIndex, (*c));
        }
        else
        {
            c2->find(depotNo)->second.push_back((*c));
        }
    }
    
    
    
    
    for(std::vector<Customer*>::iterator c = custListR2.begin(); c != custListR2.end(); c++)
    {
        float prob = rand()/float(RAND_MAX);
        
        upRouteMap cRoutes1 = deconstructRoutes(c1);
        int chromoIndex = 0;
        auto& routeIt = cRoutes1->find(depotNo)->second;
        
        std::vector<std::tuple<bool, int, float, int>> feasableList;
        /*
         Todo reserve
         */
        for(routeVec::iterator rVecIt = routeIt.begin(); rVecIt != routeIt.end(); rVecIt++)
        {
            std::vector<std::tuple<bool, int, float, int>> &&res = (*rVecIt)->testInsertions(*c, &chromoIndex);
            feasableList.insert(feasableList.end(), res.begin(), res.end());
            
        }
        
        std::sort(feasableList.begin(), feasableList.end(), [](auto& a, auto& b){ return std::get<2>(a) < std::get<2>(b);});
        
        int insertionIndex = -1;
        if(prob <= 0.8)
        {
            for(auto feasableIt = feasableList.begin(); feasableIt != feasableList.end(); feasableIt++)
            {
                if(std::get<0>(*feasableIt) == true)
                {
                    insertionIndex = std::get<3>(*feasableIt);
                    break;
                }
            }
        }
        else
        {
            if(feasableList.size() == 0)
            {
                insertionIndex = 0;
            }
            else
            {
                insertionIndex = std::get<3>(*feasableList.begin());
            }
            
        }
        
        if(insertionIndex >= 0)
        {
            if(insertionIndex >= c1->find(depotNo)->second.size())
            {
                c1->find(depotNo)->second.push_back(*c);
            }
            else
            {
                 c1->find(depotNo)->second.insert(c1->find(depotNo)->second.begin() + insertionIndex, (*c));
            }
        }
        else
        {
            c1->find(depotNo)->second.push_back((*c));
        }
    }
    
    return std::make_tuple(c1,c2);
}

spChromosome GeneticAlgorithm::mutation(spChromosome& oldChromo)
{
    float prob = rand()/float(RAND_MAX);
    if (prob < 0.50)
    {
        return swapMutation(oldChromo);
    }
    else if (prob < 0.55)
    {
        return reverseMutation(oldChromo);
    }
    else if (prob < 0.75)
    {
        return swapBestMutation(oldChromo);
    }
    
    return rerouteMutation(oldChromo);
    
}

spChromosome GeneticAlgorithm::interMutation(spChromosome& oldChromo)
{
    float prob = rand()/float(RAND_MAX);
    
    if (prob<0.00)
    {
        upRouteMap routeMap = deconstructRoutes(oldChromo);
        
        std::vector<std::tuple<int,int>> routesIndepot;
        for(auto depotIt = routeMap->begin(); depotIt != routeMap->end(); depotIt++)
        {
            routesIndepot.push_back(std::make_tuple(depotIt->first, depotIt->second.size()));
        }
        std::sort(routesIndepot.begin(), routesIndepot.end(), [](auto& a, auto& b){ return std::get<1>(a) > std::get<1>(b);});
        if (std::get<1>(routesIndepot[0]) > preferredVehicleLimit)
        {
            int to = std::get<0>(routesIndepot[routesIndepot.size()-1]);
            return insertRouteToDepot(oldChromo,std::get<0>(routesIndepot[0]),to);
        }
        
    }
    
    prob = rand()/float(RAND_MAX);
    if (prob < 0.6)
    {
        return depotSwapMutation(oldChromo);
    }
    else
    {
        return depotInsertMutation(oldChromo);
    }
    
}

spChromosome GeneticAlgorithm::swapMutation(spChromosome& oldChromo)
{
    int swapKey = getRandomChromoKey(oldChromo);
    
    upRouteMap depotRoutes = deconstructRoutes(oldChromo);
    routeVec& routes = depotRoutes->find(swapKey)->second;
    

    if((*oldChromo)[swapKey].size() < 2 || routes.size() < 2)
    {
        spChromosome newChromo = std::make_shared<Chromosome>();
        for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
        {
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
            std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
            
        }
        return newChromo;
    }

    int index0 = rand()%routes.size();
    int index1 = rand()%(routes.size()-1);
    if(index1>=index0)
    {
        index1++;
    }


    std::tuple<Customer*, int> cust0 = routes[index0]->getRandomCustomer();
    std::tuple<Customer*, int> cust1 = routes[index1]->getRandomCustomer();

    
    routes[index0]->replaceCustomer(std::get<0>(cust1), std::get<1>(cust0));
    routes[index1]->replaceCustomer(std::get<0>(cust0), std::get<1>(cust1));
    

    return reconstructChromosome(depotRoutes);
}

spChromosome GeneticAlgorithm::swapBestMutation(spChromosome& oldChromo)
{
    int swapKey = getRandomChromoKey(oldChromo);
    
    upRouteMap depotRoutes = deconstructRoutes(oldChromo);
    routeVec& routes = depotRoutes->find(swapKey)->second;
    
    
    if((*oldChromo)[swapKey].size() < 2 || routes.size() < 2)
    {
        spChromosome newChromo = std::make_shared<Chromosome>();
        for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
        {
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
            std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
            
        }
        return newChromo;
    }
    
    int index0 = rand()%routes.size();
    int index1 = rand()%(routes.size());
    if(index0 == index1)
    {
        if(routes[index0]->noRouteCustomers() <2)
        {
            index1 = (index1+1)%routes.size();
        }
    }
    
    
    Customer* cust0 = routes[index0]->popRandomCustomer();
    Customer* cust1 = routes[index1]->popRandomCustomer();
    
    routes[index0]->insertCustomerBestFeasable(cust1);
    routes[index1]->insertCustomerBestFeasable(cust0);
    
    return reconstructChromosome(depotRoutes);
}

spChromosome GeneticAlgorithm::reverseMutation(spChromosome& oldChromo)
{
    int swapKey = getRandomChromoKey(oldChromo);
    
    spChromosome newChromo = std::make_shared<Chromosome>();
    for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
    {
        (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
        std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
        
    }
    auto firstElementInDepotToReverse = (*newChromo)[newChromo->find(swapKey)->first].begin();
    size_t noElements = (*newChromo)[newChromo->find(swapKey)->first].size();
    
    int index0 = rand()%noElements;
    int index1 = rand()%(noElements-1);
    if (index1 >= index0)
    {
        index1++;
    }
    else
    {
        const int temp = index0;
        index0 = index1;
        index1 = temp;
    }
    
    std::reverse(firstElementInDepotToReverse+index0, firstElementInDepotToReverse+index1);
    
    
    return newChromo;
}

spChromosome GeneticAlgorithm::rerouteMutation(spChromosome& oldChromo)
{
    int swapKey = getRandomChromoKey(oldChromo);
    
    spChromosome newChromo = std::make_shared<Chromosome>();
    Customer* rerouted = NULL;
    for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
    {
        if(it->first == swapKey)
        {
            int rerouteIndex = rand()%it->second.size();
            rerouted = it->second[rerouteIndex];
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size()-1);
            std::copy(it->second.begin(), it->second.begin()+rerouteIndex, (*newChromo)[it->first].begin());
            std::copy(it->second.begin()+rerouteIndex + 1, it->second.end(), (*newChromo)[it->first].begin() + rerouteIndex);
            
        }
        else
        {
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
            std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
        }
        

    }
    if(rerouted == NULL)
    {
        return newChromo;
    }
    
    return insertBestFeasable(newChromo, swapKey,  rerouted);
}

spChromosome GeneticAlgorithm::insertBestFeasable(spChromosome chromo, int key, Customer* cust)
{
    upRouteMap cRoutes = deconstructRoutes(chromo);
    auto& routeIt = cRoutes->find(key)->second;
    int chromoIndex = 0;
    std::vector<std::tuple<bool, int, float, int>> feasableList;
    
    for(routeVec::iterator rVecIt = routeIt.begin(); rVecIt != routeIt.end(); rVecIt++)
    {
        std::vector<std::tuple<bool, int, float, int>> &&res = (*rVecIt)->testInsertions(cust, &chromoIndex);
        feasableList.insert(feasableList.end(), res.begin(), res.end());
        
    }
    
    std::sort(feasableList.begin(), feasableList.end(), [](auto& a, auto& b){ return std::get<2>(a) < std::get<2>(b);});
    
    int insertionIndex = -1;
    
    for(auto feasableIt = feasableList.begin(); feasableIt != feasableList.end(); feasableIt++)
    {
        if(std::get<0>(*feasableIt) == true)
        {
            insertionIndex = std::get<3>(*feasableIt);
            break;
        }
    }
    
    if (insertionIndex == -1)
    {
        if(feasableList.empty())
        {
            insertionIndex = 0;
        }
        else
        {
            insertionIndex = std::get<3>(*feasableList.begin());
        }
        
    }
    chromo->find(key)->second.insert(chromo->find(key)->second.begin() + insertionIndex, cust);
 
    return chromo;
}


spChromosome GeneticAlgorithm::depotInsertMutation(spChromosome& oldChromo)
{
    
    auto it = swappable.begin();
    std::advance(it, rand() % swappable.size());
    Customer* cust = &this->customers[ it->first];
    std::vector<Depot>& depots = it->second;
    int depotIndex = -1;
    
    spChromosome newChromo = std::make_shared<Chromosome>();
    for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
    {
        auto custInDepot = std::find((*oldChromo)[it->first].begin(), (*oldChromo)[it->first].end(), cust);
        
        
        if(custInDepot != (*oldChromo)[it->first].end())
        {
            depotIndex = it->first;
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size()-1);
            std::copy(it->second.begin(), custInDepot, (*newChromo)[it->first].begin());
            std::copy(custInDepot + 1, it->second.end(), (*newChromo)[it->first].begin() + std::distance(it->second.begin(), custInDepot));
        }
        else
        {
            (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
            std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
        }
        
    }
    
    std::vector<int> possibleDepots;
    for(auto depIt = depots.begin(); depIt != depots.end(); depIt++)
    {
        if(depIt->depotNumber != depotIndex)
        {
            possibleDepots.push_back(depIt->depotNumber);
        }
    }
    std::random_shuffle(possibleDepots.begin(), possibleDepots.end());
    return insertBestFeasable(newChromo, possibleDepots[0],  cust);
}

spChromosome GeneticAlgorithm::depotSwapMutation(spChromosome& oldChromo)
{
    spChromosome newChromo = std::make_shared<Chromosome>();
    for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
    {
        (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
        std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
    }
    
    int key1 = rand()%depots.size() +1;
    int key2 = rand()%(depots.size()-1) +1;
    if (key2 >= key1)
    {
        key2++;
    }
    else
    {
        int tmp = key1;
        key1= key2;
        key2=tmp;
    }
    // std::cout << "depots.size():" << depots.size() << std::endl;
    // std::cout << "key1:" << key1 << " key2:" << key2 << std::endl;
    // // std::cout << "newChromo->find(key1)->second:" << newChromo->find(key1)->second << std::endl;
    // std::cout << "newChromo->find(key1)->second.size():" << newChromo->find(key1)->second.size() << std::endl;
    // std::cout << "newChromo->find(key2)->second.size():" << newChromo->find(key2)->second.size() << std::endl;
    
    int cust1idx = rand()%newChromo->find(key1)->second.size();
    int cust2idx = rand()%newChromo->find(key2)->second.size();
    
    Customer* tmp = newChromo->find(key1)->second[cust1idx];
    newChromo->find(key1)->second[cust1idx] = newChromo->find(key2)->second[cust2idx];
    newChromo->find(key2)->second[cust2idx] = tmp;
    return newChromo;
}

spChromosome GeneticAlgorithm::insertRouteToDepot(spChromosome& oldChromo, int from, int to)
{
    spChromosome newChromo = std::make_shared<Chromosome>();
    for(auto it = oldChromo->begin(); it != oldChromo->end(); ++it)
    {
        (*newChromo)[it->first] = std::vector<Customer*>(it->second.size());
        std::copy(it->second.begin(), it->second.end(), (*newChromo)[it->first].begin());
    }
    
    upRouteMap routesMap = deconstructRoutes(newChromo);
    routeVec &routes = routesMap->find(from)->second;
    int routeIndexToMove = rand()%routes.size();
    
    std::vector<Customer*> custList = routes[routeIndexToMove]->getCustomerList();
    routes[routeIndexToMove]->clearCustomerList();
    
    newChromo = reconstructChromosome(routesMap);
    
    for (auto cust = custList.begin(); cust != custList.end(); ++cust)
    {
        newChromo =  insertBestFeasable(newChromo, to,  *cust);
        
    }
    return newChromo;
    
}


int GeneticAlgorithm::getRandomChromoKey(spChromosome& chromo)
{
    std::vector<int> keys;
    for(auto it = chromo->begin(); it != chromo->end(); it++)
    {
        keys.push_back(it->first);
        // std::cout << it->first;// 1,2,3,4
    }
    // std::cout << keys.size();// 4

    return keys[rand()%keys.size()];
}

spChromosome GeneticAlgorithm::reconstructChromosome(upRouteMap& routes)
{
    spChromosome newChromo(new Chromosome);
    for(auto depotIt = routes->begin(); depotIt != routes->end(); depotIt++)
    {
        (*newChromo)[depotIt->first] = std::vector<Customer*>();
        for (auto routeIt = depotIt->second.begin(); routeIt != depotIt->second.end(); routeIt++)
        {
            auto clst = (*routeIt)->getCustomerList();

            (*newChromo)[depotIt->first].insert((*newChromo)[depotIt->first].end(), clst.begin(), clst.end());
        }

    }

    return newChromo;
}


float GeneticAlgorithm::cost(upRouteMap& routes)
{
    float length = 0;
    for (routeMap::iterator routeMapIt = (*routes).begin(); routeMapIt != (*routes).end(); routeMapIt++)
    {
        for (auto routeIt = (*routeMapIt).second.begin(); routeIt != (*routeMapIt).second.end(); routeIt++)
        {
            length += (*(*routeIt)).routeLength();
        }

    }
    return length;
}


float GeneticAlgorithm::fitness(upRouteMap& routes, int gen)
{
    float length = 0;
    float penalties = 0;
    // route.hpp, using routeMap = std::map<int, routeVec>;
    for (routeMap::iterator routeMapIt = (*routes).begin(); routeMapIt != (*routes).end(); routeMapIt++)
    {
        if((*routeMapIt).second.size() > preferredVehicleLimit)
        {
            penalties += (*routeMapIt).second.size() - preferredVehicleLimit;
        }
        if(gen<penalizeRoutesUntil)
        {
            penalties += (*routeMapIt).second.size()*0.1; //prefer fewer routes!
        }
        
        for (auto routeIt = (*routeMapIt).second.begin(); routeIt != (*routeMapIt).second.end(); routeIt++)
        {
            length += (*(*routeIt)).routeLength();
        }
        
    }
    
    return length * pow(1.10, penalties) + mapSize * penalties; 
    
}
