#include "settings.h"

#include <cstring>
#include <iostream>

Settings::Settings()
{
    pmap_ = std::map<std::string, Parser>{
        {"--max", { parseUnsigned, &settings.max }},
        {"--min", { parseUnsigned, &settings.min }},
        {"--power", { parseUnsigned, &settings.power }},
        {"--stride", { parseUnsigned, &settings.stride }},
        {"--result-path", { parseString, &settings.path }},
    };
}

void Settings::help()
{
    std::cout << "Supported options:" << std::endl;
    for (auto& i : pmap_) {
        std::cout << "\t" << i.first << std::endl;
    }
}

bool Settings::parse(int argc, char* argv[])
{
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        help();
        return false;
    }

    int i = 1;

    while (i < argc) {
        auto iter = pmap_.find(argv[i]);
        if (iter == pmap_.end()) {
            std::cerr << "Unknown argument:" << argv[i] << std::endl;
            return false;
        }

        auto& p = iter->second;
        int parsed = p.f(p.item, argc - i, &argv[i]);

        if (parsed <= 0) {
            std::cerr << "Failed to parse argument: " << argv[i] << std::endl;
            return false;
        }

        i += parsed;
    }

    return true;
}

int Settings::parseUnsigned(ItemType& item, int argc, char* argv[])
{
    if (argc < 2)
        return -1;

    try {
        int val = std::stoi(argv[1]);
        if (val < 0)
            return -1;

        unsigned *res = std::get<unsigned*>(item);
        *res = val;
    } catch(...) {
        return -1;
    }

    return 2;
}

int Settings::parseString(ItemType& item,  int argc, char* argv[])
{
    if (argc < 2)
        return -1;

    try {
        std::string *res = std::get<std::string*>(item);
        *res = argv[1];
    } catch(...) {
        return -1;
    }

    return 2;
}
