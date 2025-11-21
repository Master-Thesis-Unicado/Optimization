/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/


#ifndef SRC_CONSTRAINT_ANALYSIS_ENERGY_BASED_H_
#define SRC_CONSTRAINT_ANALYSIS_ENERGY_BASED_H_

 /* === Includes === */
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include "../mission_strategy.h"
#include <engine/engine.h>
#include "../../include/mission_analysis/simple_mission_analysis.h"
#include "../../include/constraint_analysis/ca_parser.h"
#include "../mission_analysis.h"
#include "../../include/constraint_analysis/ca_plotting.h"
#include "../io/aircraft_xml.h"


class EnergyBased : public MissionAnalysisStrategy
{
public:
    /* === Constructors === */
    /**
     * @brief Construct a new energy based constraint analysis.
     * @param configuration The module configuration provided by the user.
     */
    explicit EnergyBased(
        const std::shared_ptr<node>& configuration,
        const std::shared_ptr<node>& polar,
        const std::shared_ptr<node>& aircraft_xml,
        const std::filesystem::path& engine_directory,
        const std::shared_ptr<node>& mission_xml,
        const std::filesystem::path& plot_dir
    ) : MissionAnalysisStrategy(configuration, polar, aircraft_xml, engine_directory, mission_xml, plot_dir) {}

    /* === Methods === */
    /**
     * @brief Finalize the initialization of the constraint analysis.
     */
    void initialize();

    void operator()(); // NOLINT runtime/references


}; // namespace design

#endif // SRC_CONSTRAINT_ANALYSIS_ENERGY_BASED_H_