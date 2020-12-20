
#include "depot.hpp"


Depot::Depot(int depotNumber, int maxDuration, int maxVehicleLoad):
depotNumber(depotNumber),　maxDuration(maxDuration),　maxVehicleLoad(maxVehicleLoad){}

void Depot::addCoordinates(int x, int y)
{
    this->x = x;
    this->y = y;
    
}

int Depot::getX()
{
    return x;
}

int Depot::getY()
{
    return y;
}

