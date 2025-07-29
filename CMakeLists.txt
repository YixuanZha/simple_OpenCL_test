cmake_minimum_required(VERSION 3.10)

project(OpenCLTest CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
    message(STATUS "Found OpenCL Headers: ${OpenCL_INCLUDE_DIRS}")

    add_executable(opencl_test main_test.cpp)

    target_include_directories(opencl_test PRIVATE ${OpenCL_INCLUDE_DIRS})

    target_link_libraries(opencl_test PRIVATE OpenCL::OpenCL)

else()
    message(FATAL_ERROR "Could not find OpenCL. Please check your installation and environment variables.")
endif()

message(STATUS "Configuration finished. You can now build the 'opencl_test' target.")