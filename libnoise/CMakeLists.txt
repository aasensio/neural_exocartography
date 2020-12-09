cmake_minimum_required(VERSION 3.2)
project(noise)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include lib files
include_directories("src/")
include_directories("lib/libnoise/noise/")
include_directories("lib/noiseutils/")

# Add all .cpp files to executable
file(GLOB_RECURSE SOURCE_FILES "lib/*.cpp")
add_executable(noise src/main.cpp ${SOURCE_FILES} src/Generator/SimpleGenerator.cpp src/Generator/SimpleGenerator.h src/Generator/ComplexGenerator.cpp src/Generator/ComplexGenerator.h src/Generator/Generator.h src/Generator/Generator.cpp)
