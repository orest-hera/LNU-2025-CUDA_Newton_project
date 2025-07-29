#include "system-info.h"

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef CFG_SOLVE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

#ifdef CFG_SOLVE_CUDA
void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") +
            cudaGetErrorName(err) + ", " + 	cudaGetErrorString(err));
    }
}
#endif

}

SystemInfo::SystemInfo(int argc, char* argv[])
{
    for (int i = 0; i < argc; ++i) {
        if (i !=0)
            cmd_ += ' ';
        cmd_ += argv[i];
    }

    auto now = std::chrono::system_clock::now();
    auto ms_since_epoch =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch());
    long long ms = ms_since_epoch.count() % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm* gmt = std::gmtime(&t);

    std::stringstream ss;

    ss << std::put_time(gmt, "%Y%m%d-%H%M%S") << std::setfill('0')
       << std::setw(3) << ms;

    timestamp_ = ss.str();
}

std::string SystemInfo::getTimeStamp() const
{
    return timestamp_;
}

void SystemInfo::dump(std::ostream& stream) const
{
    stream << "Launch time: " << timestamp_ << std::endl;
    stream << "Command: " << cmd_ << std::endl;

#ifdef CFG_SOLVE_CUDA
    dumpDeviceProps(stream);
#endif
}

#ifdef CFG_SOLVE_CUDA
void SystemInfo::dumpDeviceProps(std::ostream& stream) const
{
    int devId, devCount;
    cudaDeviceProp props;
    const int bsize = 512;
    char buf[bsize];

    checkCudaErrors(cudaGetDevice(&devId));
    checkCudaErrors(cudaGetDeviceCount(&devCount));
    checkCudaErrors(cudaGetDeviceProperties(&props, devId));

    stream << "Dev ID: " << devId << ", Dev count: " << devCount << '\n';

    std::snprintf(buf, sizeof(buf), "CUDA Compute %d.%d, Device: %s",
        props.major, props.minor, props.name);
    stream << buf << std::endl;

    std::snprintf(buf, sizeof(buf),
        "Memory: %zu bytes, Bus Width: %d bits, Clock Rate %d kHz",
        props.totalGlobalMem, props.memoryBusWidth, props.memoryClockRate);
    stream << buf << std::endl;

    stream << "MultiProcessorCount: " << props.multiProcessorCount << std::endl;
    stream << "GPU Cock Rate: " << props.clockRate << " kHz" << std::endl;
}
#endif
