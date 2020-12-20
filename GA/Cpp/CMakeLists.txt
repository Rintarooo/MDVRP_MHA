cmake_minimum_required(VERSION 3.1)
project(project_name_whatever CXX)

# activate c++14
# set(CMAKE_CXX_STANDARD 14)

# -g is debugger, -Wall is Warning, -O2 is Optimization
# https://qiita.com/shohirose/items/637f4b712893764a7ec1
# set(CMAKE_CXX_FLAGS "-O2 -std=c++14 -Wall -g")

if(UNIX)
	# activate CheckCXXCompileFlag 
	include(CheckCXXCompilerFlag)
	
	# check whether the CXX compiler supports a given flag
	CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

	if(COMPILER_SUPPORTS_CXX14)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
		message("compile C++14")
	elseif(COMPILER_SUPPORTS_CXX11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
		message("compile C++11")
	else()
		message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 or C++11 support. Use a different C++ compiler.")
	endif()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -g")
endif()

add_subdirectory(src)

# generate main.exe using main.cpp
add_executable(main.exe ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main.exe greetings)

message("compile: ${CMAKE_CXX_FLAGS}")
message("generate: ${PROJECT_SOURCE_DIR}/build/main.exe")
