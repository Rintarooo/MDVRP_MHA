#ifndef depot_hpp
#define depot_hpp
#include <map>
#include <vector>

class Depot
{
    int x, y;
    const int maxServiceDuration = 0; // Not specified in data, but required for task
public:
    Depot(int depotNumber, int maxDuration, int maxVehicleLoad);
    void addCoordinates(int x, int y);
    int getX();
    int getY();
    
    const int maxDuration, maxVehicleLoad, depotNumber;
    int getMaxServiceDuration(){return maxServiceDuration;}
    
};

using Swappable = std::map <int, std::vector<Depot>>;

#endif /* depot_hpp */

