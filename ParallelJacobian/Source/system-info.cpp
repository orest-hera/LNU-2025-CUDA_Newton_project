#include "system-info.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

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
}
