#include "system-build-info.h"

#include "version.h"

const std::string SystemBuildInfo::gitVersion = GIT_VERSION_STRING;
const std::string SystemBuildInfo::compilerCudaId = BUILD_COMPILER_CUDA_ID;
const std::string SystemBuildInfo::compilerCudaVer = BUILD_COMPILER_CUDA_VER;
const std::string SystemBuildInfo::compilerCxxId = BUILD_COMPILER_CXX_ID;
const std::string SystemBuildInfo::compilerCxxVer = BUILD_COMPILER_CXX_VER;
const std::string SystemBuildInfo::cuDssVer = BUILD_CUDSS_VERSION;
const std::string SystemBuildInfo::cudaArchs = BUILD_CUDA_ARCHITECTURES;
const std::string SystemBuildInfo::buildType = BUILD_TYPE;

void SystemBuildInfo::dump(std::ostream& s)
{
    s << "Version: " << gitVersion << std::endl;
    s << "CUDA Compiler " << compilerCudaId << " " << compilerCudaVer << std::endl;
    s << "CXX Compiler " << compilerCxxId << " " << compilerCxxVer << std::endl;
    s << "cuDSS Version " << cuDssVer << std::endl;
    s << "CUDA Architectures: " << cudaArchs << std::endl;
    s << "Built type: " << buildType << std::endl;
}
