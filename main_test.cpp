#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

void checkError(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << " (err_code: " << err << ")" << std::endl;
        exit(1);
    }
}

int main() {
    cl_int err;

    // === 1. OpenCL Environment Setup (Corrected Version) ===
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) { std::cerr << "No OpenCL platforms found." << std::endl; return -1; }
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    cl_platform_id platform = platforms[0];

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (num_devices == 0) { std::cerr << "No GPU devices found." << std::endl; return -1; }
    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Context creation");
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    checkError(err, "Command queue creation");
    std::cout << "OpenCL environment initialized." << std::endl;

    // === 2. Prepare Data and Buffers ===
    const int LIST_SIZE = 1024;
    std::vector<int> h_a(LIST_SIZE), h_b(LIST_SIZE), h_c(LIST_SIZE);
    for (int i = 0; i < LIST_SIZE; i++) { h_a[i] = i; h_b[i] = LIST_SIZE - i; }

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * LIST_SIZE, h_a.data(), &err);
    checkError(err, "Buffer d_a creation");
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * LIST_SIZE, h_b.data(), &err);
    checkError(err, "Buffer d_b creation");
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * LIST_SIZE, NULL, &err);
    checkError(err, "Buffer d_c creation");
    std::cout << "Data and buffers prepared." << std::endl;

    // === 3. Compile the Kernel ===
    std::ifstream kernel_file("test_kernel.cl");
    std::string source_str(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    const char *source = source_str.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    checkError(err, "Program creation");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Kernel build error: " << log.data() << std::endl; exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    checkError(err, "Kernel creation");
    std::cout << "Kernel compiled." << std::endl;

    // === 4. Execute the Kernel ===
    // *** Added error checks for each call ***
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    checkError(err, "Setting kernel argument 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    checkError(err, "Setting kernel argument 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    checkError(err, "Setting kernel argument 2");

    size_t global_work_size = LIST_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    checkError(err, "Enqueuing kernel");
    
    // It's good practice to finish before reading, to ensure execution is complete.
    err = clFinish(queue);
    checkError(err, "Finishing queue");
    std::cout << "Kernel executed." << std::endl;

    // === 5. Read Results and Verify ===
    err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, sizeof(int) * LIST_SIZE, h_c.data(), 0, NULL, NULL);
    checkError(err, "Reading back buffer d_c");

    // ... Verification loop remains the same ...
    bool success = true;
    for (int i = 0; i < 10; i++) { 
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
        }
    }
    if(success) std::cout << "Test PASSED!" << std::endl;
    else std::cout << "Test FAILED!" << std::endl;


    // === 6. Clean Up ===
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}