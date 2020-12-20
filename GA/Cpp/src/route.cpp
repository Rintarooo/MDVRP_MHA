#include "route.hpp"
#include <cmath>
#include <cstdlib>
#include "types.h"
#include <sstream>

float euclidDistance(float x1, float y1, float x2, float y2)
{
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}

Route::Route(Depot* depot, float length, int load, int serviceDuration, std::shared_ptr<std::vector<Customer*>> customerList):
depot(depot), length(length), load(load), serviceDuration(serviceDuration), customerList(customerList){}


float Route::euclidDistance(int x1, int y1, int x2, int y2)
{
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}

void Route::removeCustomers(std::vector<Customer*>& deleteList)
{
    customerList->erase( remove_if( begin(*customerList),end(*customerList),
    [&](auto x){return find(begin(deleteList),end(deleteList),x)!=end(deleteList);}), end(*customerList) );
    
    std::tuple<bool, int, float> res= checkConstraints(customerList);
    load = std::get<1>(res);
    length = std::get<2>(res);
}
#include <iostream>
std::tuple<Customer*, int> Route::getRandomCustomer()
{
    int removeIndex = rand() % customerList->size();
    Customer* cust = (*customerList)[removeIndex];
    return std::make_tuple(cust, removeIndex);

}

float Route::calculateInsertionCost(int index, Customer* cust)
{
    int prevX, prevY, nextX, nextY;
    if(index <= 0)
    {
        prevX = depot->getX();
        prevY = depot->getY();
    }
    else
    {
        prevX = (*customerList)[index-1]->x;
        prevY = (*customerList)[index-1]->y;
    }
    
    if(index < customerList->size())
    {
        nextX = (*customerList)[index]->x;
        nextY = (*customerList)[index]->y;
    }
    else
    {
        nextX = depot->getX();
        nextY = depot->getY();
    }
    float newCost = sqrt(pow((prevX-cust->x),2) + pow((prevY-cust->y),2)) + sqrt(pow((nextX-cust->x),2) + pow((nextY-cust->y),2));
    float oldCost = sqrt(pow((prevX-nextX),2) + pow((prevY-nextY),2));
    
    return newCost - oldCost;
}
Customer* Route::popRandomCustomer()
{
    std::tuple<Customer*, int> custRem = getRandomCustomer();
    
    int index = std::get<1>(custRem);
    
    int prevX, prevY, nextX, nextY;
    if(index <= 0)
    {
        prevX = depot->getX();
        prevY = depot->getY();
    }
    else
    {
        prevX = (*customerList)[index-1]->x;
        prevY = (*customerList)[index-1]->y;
    }
    
    if(index < customerList->size())
    {
        nextX = (*customerList)[index]->x;
        nextY = (*customerList)[index]->y;
    }
    else
    {
        nextX = depot->getX();
        nextY = depot->getY();
    }
    Customer* cust = std::get<0>(custRem);
    float newCost = sqrt(pow((prevX-cust->x),2) + pow((prevY-cust->y),2)) + sqrt(pow((nextX-cust->x),2) + pow((nextY-cust->y),2));
    float oldCost = sqrt(pow((prevX-nextX),2) + pow((prevY-nextY),2));

    load -= std::get<0>(custRem)->demand;
    length += newCost - oldCost;
    serviceDuration -= std::get<0>(custRem)->serviceDuration;
    customerList->erase( remove_if( begin(*customerList),end(*customerList),
    [&](auto x){return x==std::get<0>(custRem);}), end(*customerList) );
    
    return std::get<0>(custRem);
}


size_t Route::noRouteCustomers()
{
    return customerList->size();
}

float Route::routeLength()
{
    return length;
}
int Route::getRouteLoad()
{
    return load;
}

void Route::popCustomer(std::tuple<float,int> costs)
{
    length -= std::get<0>(costs);
    load -=std::get<1>(costs);
    
    customerList->pop_back();
}

void Route::insertCustomerAtBeginning(Customer* cust, std::tuple<float,int> costs)
{
    length += std::get<0>(costs);
    load +=std::get<1>(costs);
    customerList->insert(customerList->begin(), cust);
}

Customer* Route::peekLast()
{
    return (*customerList)[customerList->size()-1];
}
std::string Route::getCustomerString()
{
    if (customerList->size() == 0)
    {
        return "";
    }
    
    std::ostringstream mainstream;
    mainstream << "0 ";
    for (auto customer = customerList->begin(); customer != customerList->end(); customer++)
    {
        mainstream << (*customer)->customerNumber << " ";
    }
    mainstream <<"0";
    return mainstream.str();
    
}

float Route::calculateTotalLength()
{
    return std::get<2>(checkConstraints(customerList));
}


std::tuple<bool, int, float> Route::checkConstraints(std::shared_ptr<std::vector<Customer*>>& custList)
{
    float routeLength = 0.0;
    int routeLoad = 0;
    
    int other_x = this->depot->getX();
    int other_y = this->depot->getY();
    
    for(auto customerIt = custList->begin(); customerIt != custList->end(); ++customerIt)
    {
        routeLength += this->euclidDistance((*customerIt)->x, (*customerIt)->y, other_x, other_y);
        routeLoad += (*customerIt)->demand;
        other_x = (*customerIt)->x;
        other_y = (*customerIt)->y;
    }
    routeLength += this->euclidDistance(this->depot->getX(), this->depot->getY(), other_x, other_y);
    bool validRoute = (this->depot->maxDuration == 0 || routeLength <= this->depot->maxDuration) && routeLoad <= this->depot->maxVehicleLoad;
    
    return std::make_tuple(validRoute, routeLoad, routeLength);
}
std::tuple<bool, int, float> Route::checkConstraints()
{
    return checkConstraints(this->customerList);
}


std::tuple<bool, int, float> Route::checkConstraints(int index, Customer* cust)
{
    float newRouteLength = length;
    float newRouteLoad = load;
    float newRouteServiceDuration = serviceDuration;
    int prevX, prevY, nextX, nextY;
    if(index <= 0)
    {
        prevX = depot->getX();
        prevY = depot->getY();
    }
    else
    {
        prevX = (*customerList)[index-1]->x;
        prevY = (*customerList)[index-1]->y;
    }
    
    if(index < customerList->size())
    {
        nextX = (*customerList)[index]->x;
        nextY = (*customerList)[index]->y;
    }
    else
    {
        nextX = depot->getX();
        nextY = depot->getY();
        
    }

    
    
    
    float newCost = sqrt(pow((prevX-cust->x),2) + pow((prevY-cust->y),2)) + sqrt(pow((nextX-cust->x),2) + pow((nextY-cust->y),2));
    float oldCost = sqrt(pow((prevX-nextX),2) + pow((prevY-nextY),2));
    
    newRouteLength += newCost - oldCost;
    newRouteLoad += cust->demand;
    newRouteServiceDuration += cust->serviceDuration;
    
    bool validRoute = (this->depot->maxDuration == 0 || newRouteLength <= this->depot->maxDuration) && newRouteLoad <= this->depot->maxVehicleLoad && (this->depot->getMaxServiceDuration() == 0 || newRouteServiceDuration <= this->depot->getMaxServiceDuration());
    
    return std::make_tuple(validRoute, cust->demand, newCost - oldCost);
    
}

std::tuple<float,int> Route::removeLastGain()
{
    if(customerList->size() == 1)
    {
        Customer* cust = (*customerList)[0];
        return std::make_tuple(2*euclidDistance(cust->x, cust->y, depot->getX(), depot->getY()), cust->demand);
    }
    Customer* removeCust = (*customerList)[customerList->size()-1];
    Customer* newBackCust = (*customerList)[customerList->size()-2];
    
    
    float oldCost = euclidDistance(newBackCust->x, newBackCust->y, removeCust->x, removeCust->y) + euclidDistance( depot->getX(), depot->getY(), removeCust->x, removeCust->y);
    float newCost = euclidDistance(newBackCust->x, newBackCust->y, depot->getX(), depot->getY());
    
    return std::make_tuple(oldCost-newCost,removeCust->demand);
}

void Route::replaceCustomer(Customer* cust, int index)
{
    
    Customer* oldCust = (*this->customerList)[index];
    (*this->customerList)[index] = cust;
    
    //Added to fix loop problem
    int prevX, prevY, nextX, nextY;
    if(index < 1)
    {
        prevX = depot->getX();
        prevY = depot->getY();
    }
    else
    {
        prevX = (*customerList)[index-1]->x;
        prevY = (*customerList)[index-1]->y;
    }
    
    if(index < customerList->size()-1)
    {
        nextX = (*customerList)[index+1]->x;
        nextY = (*customerList)[index+1]->y;
    }
    else
    {
        nextX = depot->getX();
        nextY = depot->getY();
    }
    
    float newCost = sqrt(pow((prevX-cust->x),2) + pow((prevY-cust->y),2)) + sqrt(pow((nextX-cust->x),2) + pow((nextY-cust->y),2));
    float oldCost = sqrt(pow((prevX-oldCust->x),2) + pow((prevY-oldCust->y),2)) + sqrt(pow((nextX-oldCust->x),2) + pow((nextY-oldCust->y),2));
    
    length += newCost - oldCost;
    load += cust->demand - oldCust->demand;
    load += cust->serviceDuration - oldCust->serviceDuration;
    
    
}
void Route::clearCustomerList()
{
    customerList = std::make_shared<std::vector<Customer*>>();
    length = 0;
    load = 0;
    serviceDuration = 0;
}

std::vector<Customer*> Route::getCustomerList()
{
    return {(*customerList)};
}

void Route::insertCustomer(Customer* cust, int index)
{
    load += cust->demand;
    serviceDuration += cust->serviceDuration;
    length += calculateInsertionCost(index, cust);
    customerList->insert(customerList->begin()+index, cust);
}

void Route::insertCustomerBestFeasable(Customer* cust)
{
    
    int chromoIndex = 0;
    std::vector<std::tuple<bool, int, float, int>> feasableList = testInsertions(cust, &chromoIndex);
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
    
    
    insertCustomer(cust, insertionIndex);
}

std::vector<std::tuple<bool, int, float, int>> Route::testInsertions(Customer* cust, int* chromoIndex)
{
    std::vector<std::tuple<bool, int, float, int>> results;
    for(int i = 0; i < customerList->size(); i++)
    {
  
        std::tuple<bool, int, float> checkRes = checkConstraints(i, cust);
        results.push_back(std::make_tuple(std::get<0>(checkRes),std::get<1>(checkRes), std::get<2>(checkRes), (*chromoIndex)++));
    }
    return results;
}


