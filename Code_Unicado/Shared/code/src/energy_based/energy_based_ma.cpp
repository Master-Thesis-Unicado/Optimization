/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#include "energy_based_ma.h"
#include <algorithm>
#include <filesystem>
#include <format>
#include <list>
#include <ranges>
#include <cmath>
#include <vector>
#include <numbers>
#include <iostream>
#include <string>
#include <moduleBasics/module.h>
#include <engine/engine.h>
#include "../../include/constraint_analysis/ca_parser.h"
#include "../mission_analysis.h"
#include "../io/aircraft_xml.h"
#include "../../include/mission_analysis/simple_mission_analysis.h"
#include "../../include/avl/avl_interface.h"


void EnergyBased::initialize()
{
    if (!this->aircraft_xml()) {
        throw std::runtime_error("[EnergyBased::initialize()] aircraft_xml is null!");
    };
    if (!this->configuration()) {
        throw std::runtime_error("[EnergyBased::initialize()] config_xml is null!");
    };
    if (!this->polar()) {
        throw std::runtime_error("[EnergyBased::initialize()] polar_xml is null!");
    };
    avl_interface::geometry_interface_avl(this->aircraft_xml());
    std::vector<double> M_list = { 0.3, 0.5, 0.7, 0.72, 0.74, 0.76, 0.78 };
    std::vector<double> density_list = { 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> CL_lims = { 0.0, 1.5 };
    avl_interface::create_polar(M_list, density_list, CL_lims);
}

void EnergyBased::operator()()
{
    std::vector<Engine> engines = {};
    auto ac_engines = this->aircraft_xml()->getVector("aircraft_exchange_file/component_design/propulsion/specific/propulsion");
    for (auto ac_engine : ac_engines)
    {
        std::string engine_model = ac_engine->at("/engine/model/value");
        double scale_factor = ac_engine->at("scale_factor/value");
        Engine engine = Engine(this->engine_directory() / engine_model, scale_factor);
        engines.push_back(engine);
    };
    std::vector<double> thrust_share = {};
    auto propulsors = this->aircraft_xml()->getVector("aircraft_exchange_file/requirements_and_specifications/design_specification/propulsion/propulsor");
    for (auto propulsor : propulsors)
    {
        double propulsor_thrust_share = propulsor->at("/thrust_share/value");
        thrust_share.push_back(propulsor_thrust_share);
    };
    auto mission = this->mission_xml()->at("/mission");
    Mission simple_mission(this->polar(), std::make_shared<node>(mission), engines, thrust_share);
}
