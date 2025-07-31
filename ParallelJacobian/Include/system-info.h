#pragma once

#include <ostream>
#include <string>

class SystemInfo
{
public:
    SystemInfo(int argc, char* argv[]);

    void dump(std::ostream&) const;
    void dump_resource_usage(std::ostream&) const;
    std::string getTimeStamp() const;

private:
#ifdef CFG_SOLVE_CUDA
    void dumpDeviceProps(std::ostream& stream) const;
#endif

    std::string cmd_;
    std::string timestamp_;
};
