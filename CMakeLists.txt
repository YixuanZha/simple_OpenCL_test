# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(OpenCLTest CXX)

# 设置C++标准为C++11或更高版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCL包。这是CMake的标准做法。
# 它会定义OpenCL_FOUND, OpenCL_INCLUDE_DIRS 和 OpenCL_LIBRARIES等变量。
find_package(OpenCL REQUIRED)

# 如果成功找到了OpenCL
if(OpenCL_FOUND)
    # 打印找到的OpenCL头文件和库的路径，方便调试
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
    message(STATUS "Found OpenCL Headers: ${OpenCL_INCLUDE_DIRS}")

    # 添加你的源文件来生成可执行文件
    add_executable(opencl_test main_test.cpp)

    # 将OpenCL的头文件目录添加到你的项目中
    target_include_directories(opencl_test PRIVATE ${OpenCL_INCLUDE_DIRS})

    # 将OpenCL库链接到你的可执行文件
    # OpenCL::OpenCL 是一个导入的目标，由 find_package 提供
    target_link_libraries(opencl_test PRIVATE OpenCL::OpenCL)

else()
    # 如果没有找到OpenCL，打印错误信息
    message(FATAL_ERROR "Could not find OpenCL. Please check your installation and environment variables.")
endif()

# 打印一条消息，指示CMake配置完成
message(STATUS "Configuration finished. You can now build the 'opencl_test' target.")