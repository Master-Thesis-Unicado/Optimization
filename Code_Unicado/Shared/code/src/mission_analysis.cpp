/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#include <cmath>
#include <vector>
#include <numbers>
#include <iostream>
#include <string>
#include <algorithm>
#include <format>
#include <variant>
#include <moduleBasics/module.h>
#include <engine/engine.h>
#include "io/aircraft_xml.h"
#include "mission_analysis.h"
#include "energy_based/energy_based_ma.h"
#include "../include/geometry_interface/CPACS/wing_interface.h"
#include "../include/avl/avl_geometry_interface.h"
#include "../include/lifting_line/lifting_line_geometry_interface.h"
#include "../include/lifting_line/lifting_line_interface.h"

void DefaultAnalysis::initialize() {
    /* Load XML resources */
    this->aircraft_xml = aixml::openDocument(this->get_RuntimeIO()->acxmlAccess);
    this->configuration_xml = aixml::openDocument(this->get_RuntimeIO()->moduleConfAccess);
    std::filesystem::path file = this->get_RuntimeIO()->acxmlAccess;
    this->engine_directory = file.parent_path() / "engine_data";
    std::filesystem::path confFile = this->get_RuntimeIO()->moduleConfAccess;
    std::filesystem::path missionFile = confFile.parent_path() / "simple_mission.xml";
    this->mission_xml = aixml::openDocument(missionFile);

    // TEST FOR AVL
    std::string polar_file = this->aircraft_xml->at("aircraft_exchange_file/analysis/aerodynamics/polar/polar_file/value");
    std::filesystem::path polarFile = file.parent_path() / "aero_data" / polar_file;
    // polarFile = confFile.parent_path() / "include/avl/polar.xml";
    auto polar_xml = aixml::openDocument(polarFile);
    std::filesystem::path plot_dir = this->get_RuntimeIO()->getPlotDir();

    //// TEST FOR LIFTING LINE
    //auto aircraft_geometry = avl_geo_interface::create_aircraft_geometry(this->aircraft_xml);
    //auto lili_conf = aixml::openDocument(confFile.parent_path() / "include/lifting_line/lili_conf.xml");
    //lifting_line_interface::create_input_for_lifting_surfaces(this->aircraft_xml, aircraft_geometry, lili_conf->at("case"));
    //lifting_line_interface::exec_lifting_line("main_wing");
    //auto output_xml = aixml::openDocument(confFile.parent_path() / "include/lifting_line/main_wing/main_wing.lili.V3.1/export/main_wing.xml");
    //auto polar = lifting_line_interface::read_lifting_line_output(output_xml, 0.);
    //polar.visualize();

    //// TEST FOR CPACS
    //auto CPACS_xml = aixml::openDocument(confFile.parent_path() / "include/geometry_interface/CPACS/CPACS_Wing.xml");
    //CPACS::Wing main_wing = CPACS::Wing(CPACS_xml->at("cpacs/vehicles/aircraft/model/wings/wing"));
    //auto airfoils = CPACS_xml->getVector("cpacs/vehicles/profiles/wingAirfoils/wingAirfoil");
    //std::for_each(airfoils.begin(), airfoils.end(),
    //    [](auto& airfoil)
    //    {
    //        CPACS::write_airfoil(*airfoil);
    //    });
    //std::fstream wing_acXML;
    //wing_acXML.open(confFile.parent_path() / "include/geometry_interface/CPACS/new_wing.xml", std::ios::out);
    //main_wing.to_acXML(wing_acXML);
    //auto wing_geometry = main_wing.create_geometry();
    //std::fstream mesh_file;
    //mesh_file.open("wing_mesh.stl", std::ios::out);
    //CGAL::IO::write_STL(mesh_file, geom2::transform::to_mesh(wing_geometry));
    //mesh_file.close();

    /* Select the constraint analysis strategy*/
    const std::string method(this->configuration_xml->at("module_configuration_file/program_settings/method/value"));
    if (method == "Energy_Based")
        this->mission_analyzer = std::make_unique<EnergyBased>(
            this->configuration_xml,
            polar_xml,
            this->aircraft_xml,
            this->engine_directory,
            this->mission_xml,
            plot_dir);
    else
        throw std::runtime_error("[Constraint_Analysis] The mission analysis method '" + method + "' is not supported.");

    this->mission_analyzer->initialize();
}

void DefaultAnalysis::run() {
    (*this->mission_analyzer)();
}

void DefaultAnalysis::update() {
}

void DefaultAnalysis::report() {
}

void DefaultAnalysis::save() {
}
