#ifndef filereader_hpp
#define filereader_hpp
#include <iostream>
#include <fstream>// ifstream
#include <sstream>// istringstream
#include <vector>
#include <string>
#include "depot.hpp"
#include "customer.hpp"

int readFile(const char*, std::vector<Depot>&, std::vector<Customer>&, int []);

#endif