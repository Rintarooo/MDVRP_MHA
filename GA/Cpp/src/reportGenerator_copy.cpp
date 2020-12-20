#include "reportGenerator.hpp"
#include <sstream>
#include <iomanip>

std::string generateReport(upRouteMap& routes)
{
    std::ostringstream mainstream;
    std::string report = "";
    
    float length = 0;
    for (routeMap::iterator routeMapIt = (*routes).begin(); routeMapIt != (*routes).end(); routeMapIt++)
    {
        int routeInDepot = 1;
        for (auto routeIt = (*routeMapIt).second.begin(); routeIt != (*routeMapIt).second.end(); routeIt++)
        {
            std::ostringstream linestream;
            linestream << (*routeMapIt).first << "   "<< routeInDepot++ << "   "  << std::fixed << std::setw( 8 ) << std::setprecision( 2 ) <<(*routeIt)->routeLength()<< "   " <<
            (*routeIt)->getRouteLoad() << "   " <<(*routeIt)->getCustomerString()<<"\r\n";
            length += (*routeIt)->routeLength();
            report = report + linestream.str();
        }
        
    }
    mainstream << length << "\r\n"<<report;
    return mainstream.str();
    
}
