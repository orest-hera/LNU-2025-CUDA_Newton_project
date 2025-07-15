#include "report.h"

#include <filesystem>
#include <iostream>
#include <system_error>

namespace fs = std::filesystem;

namespace Report {

bool createReportDir(std::string path)
{
    if (fs::exists(path)) {
        std::cerr << "Path already exists: " << path << std::endl;
        return false;
    }

    bool ret = false;

    try {
        ret = fs::create_directory(path);
    } catch (std::system_error const& e) {
        std::cerr << e.what() << std::endl;

        return false;
    }

    return ret;
}

} // namespace Report
