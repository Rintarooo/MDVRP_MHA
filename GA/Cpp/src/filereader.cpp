#include "filereader.hpp"

int readFile(const char* filename, std::vector<Depot> &depots, std::vector<Customer> &customers, int scale[])//(const std::string filename)
{
	 std::ifstream inFile(filename, std::ios::in);
   if (!inFile.is_open()) {
   	std::cerr << "Error: cannot open file: " << filename;
   	std::exit(1);// return -1;
   }
    
   std::string line;
   std::getline(inFile, line);// read 1 line string from stream class
   std::istringstream is(line);// 1 line string into stream class for extract only
    
   // int _, m, n, t;
   // is >> _ >> m >> n >> t;
   int m, n, t;
   is >> m >> n >> t;
   is.clear();
   is.seekg(0);
     
   for(int i = 1; i <= t; ++i)
   {
        std::getline(inFile, line);
        std::istringstream is2(line);
        int D, Q;
        is2 >> D >> Q;
        depots.push_back(Depot(i, D, Q));
   }
   
    
   for(int _ = 0; _< n; _++)
   {
        int i, x, y, d, q;
        
        std::getline(inFile, line);
        std::istringstream is2(line);
        is2 >> i >> x >> y >> d >> q;        
        customers.push_back(Customer(i, x, y, d, q));
        
        if (x< scale[0])
        {
            scale[0] = x;
        }
        if (y< scale[1])
        {
            scale[1] = y;
        }
        if (x> scale[2])
        {
            scale[2] = x;
        }
        if (y> scale[3])
        {
            scale[3] = y;
        }
   }
    
   for(int i = 0; i < t; i++)
   {
        int x, y, dc;
        std::getline(inFile, line);
        std::istringstream is2(line);
        is2 >> dc >> x >> y ;
        depots[i].addCoordinates(x, y);
        
        if (x< scale[0])
        {
            scale[0] = x;
        }
        if (y< scale[1])
        {
            scale[1] = y;
        }
        if (x> scale[2])
        {
            scale[2] = x;
        }
        if (y> scale[3])
        {
            scale[3] = y;
        }
   }
   return m;
}   