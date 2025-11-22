/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#include <cmath>
#include <vector>
#include <numbers>
#include <memory>
#include <moduleBasics/module.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "mission_strategy.h"

#ifndef SRC_MISSION_ANALYSIS_H_
#define SRC_MISSION_ANALYSIS_H_


class DefaultAnalysis : public Module {
public:

    DefaultAnalysis(const int argc, char** argv, const std::string& toolName, const std::string& toolVersion)
        : Module(argc, argv, toolName, toolVersion)
    {
    }

    DefaultAnalysis(const std::string& toolName, const std::string& toolVersion, const std::filesystem::path& rtConfigXML)
        : Module(toolName, toolVersion, rtConfigXML)
    {
    }

    ~DefaultAnalysis() = default;

    void initialize() override;
    void run() override;
    void update() override;
    void report() override;
    void save() override;

private:

    std::unique_ptr<MissionAnalysisStrategy> mission_analyzer;
    std::shared_ptr<node> configuration_xml;
    std::shared_ptr<node> aircraft_xml;
    std::filesystem::path engine_directory;
    std::shared_ptr<node> mission_xml;
};

#endif