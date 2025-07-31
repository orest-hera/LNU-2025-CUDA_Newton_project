#pragma once

#include <memory>
#include <ostream>
#include <string>

#ifdef CFG_SOLVE_CUDA
class GpuMonitor;
#endif

class SystemInfo
{
public:
    SystemInfo(int argc, char* argv[]);
    ~SystemInfo();

    void dump(std::ostream&) const;
    void dump_resource_usage(std::ostream&) const;
    std::string getTimeStamp() const;

private:
#ifdef CFG_SOLVE_CUDA
    void dumpDeviceProps(std::ostream& stream) const;

    std::unique_ptr<GpuMonitor> gpu_mon_;
#endif

    std::string cmd_;
    std::string timestamp_;
};
