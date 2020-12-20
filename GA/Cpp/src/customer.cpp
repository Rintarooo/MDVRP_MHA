
#include "customer.hpp"


Customer::Customer(int customerNumber, int x, int y, int serviceDuration, int demand):
customerNumber(customerNumber),　x(x),　y(y),　serviceDuration(serviceDuration),　demand(demand){}

bool Customer::operator==(const Customer& other) const
{
    return this->customerNumber == other.customerNumber;
}
