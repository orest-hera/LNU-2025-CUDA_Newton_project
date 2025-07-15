#pragma once

#include <ostream>
#include <string>

struct SystemBuildInfo
{
    static const std::string gitVersion;
    static const std::string compilerCudaId;
    static const std::string compilerCudaVer;
    static const std::string compilerCxxId;
    static const std::string compilerCxxVer;
    static const std::string cuDssVer;
    static const std::string cudaArchs;
    static const std::string buildType;

    static void dump(std::ostream&);
};
