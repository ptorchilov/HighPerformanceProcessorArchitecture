#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ppm_helpers.h"

__global__ void addKernel()
{

}

int main()
{
    char* path = "D:\\work\\VS\\sem8\\avmis\\HighPerformanceProcessorArchitecture\\images\\sample.ppm";
    char* test = "D:\\work\\VS\\sem8\\avmis\\HighPerformanceProcessorArchitecture\\images\\test.ppm";

    uint8_t* data;

    unsigned int w = 3000, h = 3000;

    load_ppm_alpha(path, &data, &w, &h);

    store_ppm_alpha(test, data, w, h);

    return 0;
}