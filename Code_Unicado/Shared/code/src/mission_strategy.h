/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef SRC_MA_STRATEGY_H_
#define SRC_MA_STRATEGY_H_

 /* === Includes === */
#include <memory>
#include <moduleBasics/strategySelector.h>
#include <aixml/node.h>
#include <moduleBasics/strategySelector.h>

/* === Classes === */
/**
 * @class ConstraintAnalysisStrategy
 * @brief The specialization of the strategy class for the constraint_analysis tool.
 */
class MissionAnalysisStrategy : public Strategy
{
public:
    /* === Constructors === */
    explicit MissionAnalysisStrategy(
        const std::shared_ptr<node>& configuration,
        const std::shared_ptr<node>& polar,
        const std::shared_ptr<node>& aircraft_xml,
        const std::filesystem::path& engine_directory,
        const std::shared_ptr<node>& mission_xml,
        const std::filesystem::path& plot_dir) : configuration_(configuration), polar_(polar), aircraft_xml_(aircraft_xml), engine_directory_(engine_directory), mission_xml_(mission_xml), plot_dir_(plot_dir) {}

    /**
     * @brief Destroy the PropulsionStrategy instance.
     */
    ~MissionAnalysisStrategy() override = default;

    /* === Populate the strategy template methods === */
    void initialize() override {};
    void run() override {};
    void update() override {};
    void report() override {};
    void save() override {};

    virtual void operator()() // NOLINT runtime/reference
    {
        throw std::invalid_argument("The strategy does not yet work.");
    };

    /* === Methods === */
    /**
     * @brief Get the configuration of the module.
     *
     * @return std::shared_ptr<node> The configuration of the module.
     */
    [[nodiscard]] auto configuration() const -> const std::shared_ptr<node> { return configuration_; }

    [[nodiscard]] auto polar() const -> const std::shared_ptr<node> { return polar_; }

    [[nodiscard]] auto aircraft_xml() const -> const std::shared_ptr<node> { return aircraft_xml_; }

    [[nodiscard]] auto engine_directory() const -> const std::filesystem::path { return engine_directory_; }

    [[nodiscard]] auto mission_xml() const -> const std::shared_ptr<node> { return mission_xml_; }

    [[nodiscard]] auto plot_dir() const -> const std::filesystem::path { return plot_dir_; }

private:
    /* === Properties ===*/
    std::shared_ptr<node> configuration_;               /** The configuration of the module. */
    std::shared_ptr<node> polar_;                       /** The polar file. */
    std::shared_ptr<node> aircraft_xml_;                /** The aircraft xml file. */
    const std::filesystem::path engine_directory_;      /** The engine data directory. */
    std::shared_ptr<node> mission_xml_;                 /** The mission data directory. */
    const std::filesystem::path plot_dir_;              /** The engine data directory. */
};

#endif 