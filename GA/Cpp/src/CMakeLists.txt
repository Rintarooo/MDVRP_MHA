add_library(greetings
  SHARED
    cluster.cpp
    customer.cpp
    depot.cpp
    geneticAlgorithm.cpp
    reportGenerator.cpp
    route.cpp
    filereader.cpp
  )

# PROJECT_SOURCE_DIR is absolute path of root directory for this repository
# target_include_directories is telling the path of "include" directory for a library
# here, target is main.exe. So PUBLIC tells the path to both greetings and main.exe(= target).
target_include_directories(greetings
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  )
