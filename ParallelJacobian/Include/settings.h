#pragma once

#include <functional>
#include <map>
#include <string>
#include <variant>

class Settings
{
public:
    Settings();
    bool parse(int argc, char* argv[]);

    struct SettingsData {
        unsigned max{1000};
        unsigned min{1000};
        unsigned power{3};
        unsigned stride{100};
        bool is_cpu{false};
        bool is_cublas{false};
        bool is_cudss{false};
        std::string path{"../results"};
        bool report_subdir{false};
    } settings;

private:
    typedef std::variant<std::string*, unsigned*, bool*> ItemType;

    struct Parser {
        std::function<int (ItemType&, int argc, char* argv[])> f;
        ItemType item;
    };

    static int parseUnsigned(ItemType&, int argc, char* argv[]);
    static int parseString(ItemType&, int argc, char* argv[]);
    static int parseBool(ItemType&, int argc, char* argv[]);

    void help();

    std::map<std::string, Parser> pmap_;
};
