#ifndef route_hpp
#define route_hpp
#include "depot.hpp"
#include "customer.hpp"
#include <vector>
#include <tuple>
#include <map>

class Route
{
    Depot* depot;
    std::shared_ptr<std::vector<Customer*>> customerList;
    float length;
    int load;
    int serviceDuration;
    
    float euclidDistance(int x1, int y1, int x2, int y2);
    std::tuple<bool, int, float> checkConstraints(std::shared_ptr<std::vector<Customer*>>& custList);
    std::tuple<bool, int, float> checkConstraints();
    float calculateInsertionCost(int index, Customer* cust);
    
    
    
    
    
public:
    Route(Depot* depot, float length, int load, int serviceDuration, std::shared_ptr<std::vector<Customer*>> customerList);
    void removeCustomers(std::vector<Customer*>& deleteList);
    void replaceCustomer(Customer* cust, int index);
    std::tuple<Customer*, int> getRandomCustomer();
    size_t noRouteCustomers();
    float routeLength();
    int getRouteLoad();
    std::vector<Customer*> getCustomerList();
    std::vector<std::tuple<bool, int, float, int>> testInsertions(Customer* cust, int* chromoIndex);
    void clearCustomerList();
    void popCustomer(std::tuple<float,int> costs);
    std::tuple<float,int>  removeLastGain();
    std::tuple<bool, int, float> checkConstraints(int index, Customer* cust);
    void insertCustomerAtBeginning(Customer* cust, std::tuple<float,int> costs);
    Customer* peekLast();
    Customer* popRandomCustomer();
    void insertCustomer(Customer* cust, int index);
    void insertCustomerBestFeasable(Customer* cust);
    std::string getCustomerString();
    float calculateTotalLength();
    
    
    
};

using routeVec = std::vector<std::unique_ptr<Route>>;
using routeMap = std::map<int, routeVec>;
// using routeMap = std::map<depotNo, routeVec>;
using upRouteMap = std::unique_ptr<routeMap>;

#endif /* route_hpp */
