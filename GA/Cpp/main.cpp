#include <iostream>
#include <vector>

#include "filereader.hpp"
#include "depot.hpp"
#include "customer.hpp"
#include "geneticAlgorithm.hpp"

int main(int argc, char *argv[])
{
    srand(1999);
    if (argc <= 1){
        std::cerr << "argc:" << argc << "should be >= 2";
        return -1;
    }
    
    const char* filename = argv[1];
    std::vector<Depot> depots;
    std::vector<Customer> customers;
    int scale[4] = {INT_MAX, INT_MAX, INT_MIN, INT_MIN};// INT_MAX = 2147483647(= 2^31-1), INT_MIN = -2147483647(= -2^31-1)

    int m = readFile(filename, depots, customers, scale);
    // m is available vehicle number

    GeneticAlgorithm ga(depots, customers, scale, m);
    // ga.run(5000, 5000, 400);
    ga.run(500, 500, 400);
    
    return 0;
}

