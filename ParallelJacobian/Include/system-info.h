#pragma once

#include <ostream>
#include <string>

class SystemInfo
{
public:
    SystemInfo(int argc, char* argv[]);

    void dump(std::ostream&) const;
    std::string getTimeStamp() const;

private:
    void dumpDeviceProps(std::ostream& stream) const;

    std::string cmd_;
    std::string timestamp_;
};
