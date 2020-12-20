#ifndef customer_hpp
#define customer_hpp
#include <map>
#include <vector>

class Customer
{
public:
    Customer(int customerNumber, int x, int y, int serviceDuration, int demand);
    bool operator==(const Customer& rhs) const;
    const int customerNumber, x, y, serviceDuration, demand;
};

using Chromosome = std::map <int, std::vector<Customer*>>;
// using Chromosome = std::map <depotNo, std::vector<Customer*>>;
using spChromosome = std::shared_ptr<Chromosome>;

#endif /* customer_hpp */