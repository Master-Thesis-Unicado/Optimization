#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>
#include <engine/engine.h>
#include <matplot/matplot.h>
#include <atmosphere/atmosphere.h>
//#include <energyCarriers/energyCarriers.h>
#include "../../include/mission_analysis/simple_mission_analysis.h"
#include "../../include/mission_analysis/plotter.h"


void Mission::visualize_mission()
{
    // initialize the plotting
    Canvas mission_profile("plots/mission_profile.svg");
    Canvas speed_time("plots/speed_time.svg");
    Canvas acceleration_time("plots/acceleration_time.svg");
    Canvas mass_distance("plots/mass_distance.svg");
    Canvas thrust_distance("plots/thrust_distance.svg");
    Canvas speed_distance("plots/speed_distance.svg");

    // initialize the containers
    std::vector<double> distance_traveled = {};
    std::vector<double> altitude = {};
    std::vector<double> dv_dt = {};
    std::vector<double> dh_dt = {};
    std::vector<double> time = { 0. };
    std::vector<double> v = {};
    std::vector<double> mass = {};
    std::vector<double> thrust = {};

    // fill the containers with data from the mission analysis
    for (auto seg : this->segments)
    {
        distance_traveled.push_back(seg.range / 1000.);
        altitude.push_back(seg.altitude);
        dv_dt.push_back(seg.dv_dt);
        dh_dt.push_back(seg.dh_dt);
        time.push_back(seg.delta_t / (3600.) + time.back());
        v.push_back(this->atm.getSpeedOfSound(seg.altitude) * seg.Mach);
        mass.push_back(seg.initial_weight / g0);
        thrust.push_back(seg.thrust);
    }

    mission_profile.add_data(distance_traveled, altitude);
    mission_profile.set_axis_titles("distance traveled [km]", "altitude [m]");
    mission_profile.save_canvas();

    speed_time.add_data(time, v);
    speed_time.add_data(time, dh_dt);
    speed_time.set_axis_titles("time [hr]", "speed [m/s]");
    speed_time.save_canvas();

    speed_distance.add_data(distance_traveled, v);
    speed_distance.add_data(distance_traveled, dh_dt);
    speed_distance.set_axis_titles("distance traveled [km]", "speed [m/s]");
    speed_distance.save_canvas();

    acceleration_time.add_data(time, dv_dt);
    acceleration_time.set_axis_titles("time [hr]", "acceleration [m/s^2]");
    acceleration_time.save_canvas();

    mass_distance.add_data(distance_traveled, mass);
    mass_distance.set_axis_titles("distance traveled [km]", "mass [kg]");
    mass_distance.save_canvas();

    thrust_distance.add_data(distance_traveled, thrust);
    thrust_distance.set_axis_titles("distance traveled [km]", "thrust [N]");
    thrust_distance.save_canvas();

};

//void Mission::create_csv_out()
//{
//    // setup AC .xml file
//    std::filesystem::path myPath{ "../projects2/CSMR-2020.xml" };
//    std::shared_ptr<node> aircraft_xml = aixml::openDocument(myPath);
//
//    // setup fuel properties
//    std::string energy_node{ "aircraft_exchange_file/requirements_and_specifications/design_specification/energy_carriers/energy_carrier" };
//    std::string fuel_type = aircraft_xml->at(energy_node + "/type/value");
//    double fuel_density = aircraft_xml->at(energy_node + "/density/value");
//    double fuel_energy_density = EnergyCarrier(fuel_type, fuel_density).gravimetric_energy_density;
//
//    // setup MAC
//    double mean_aerodynamic_chord{ aircraft_xml->at("aircraft_exchange_file/analysis/aerodynamics/reference_values/MAC/value") };
//
//    // initialize the containers
//    std::vector<double> total_time{ 0.0 };                                                    // in [s]
//    std::vector<double> distance_traveled{ 0.0 };                                             // in [m]
//    std::vector<double> altitude{ 0.0 };                                                      // in [m]
//    std::vector<double> flight_level{ 0.0 };                                                  // in [FL]
//    std::vector<std::string> mode_name{ "constant_cruise" };                                  // in [-]
//    std::vector<double> total_mass{ this->segments.at(0).initial_weight / g0 };               // in [kg]
//    std::vector<int> energy_carrier_ID{ aircraft_xml->at(energy_node).getIntAttrib("ID") };   // in [-]
//    std::vector<double> thrust{ 0.0 };                                                        // in [N]
//    std::vector<double> shaftpower_offtake{ 0.0 };                                            // in [W]
//    std::vector<double> bleed_air{ 0.0 };                                                     // in [kg/s]
//    std::vector<double> fuelflow{ 0.0 };                                                      // in [kg/s]
//    std::vector<double> fuel_consumed{ 0.0 };                                                 // in [kg]
//    std::vector<double> energy_consumed{ 0.0 };                                               // in [J]
//    std::vector<double> mach{ 0.0 };                                                          // in [-]
//    std::vector<double> cas{ 0.0 };                                                           // in [m/s]
//    std::vector<double> tas_in_meters_per_sec{ 0.0 };                                         // in [m/s]
//    std::vector<double> tas_in_knots{ 0.0 };                                                  // in [knots]
//    std::vector<double> roc{ 0.0 };                                                           // in [ft/min]
//    std::vector<double> sar{ 0.0 };                                                           // in [m/kg]
//    std::vector<std::string> aero_config{ "clean" };                                          // in [-]
//    std::vector<double> CL{ 0.0 };                                                            // in [-]
//    std::vector<double> spoiler_factor{ 0.0 };                                                // in [-]
//    std::vector<double> raynolds_number{ 0.0 };                                               // in [-]
//    std::vector<double> engine_rating{ 0.0 };                                                 // in [-]
//    std::vector<double> engine1_N1{ 0.0 };                                                    // in [-]
//    std::vector<double> engine2_N1{ 0.0 };                                                    // in [-]
//    std::vector<double> CoG_aircraft{ 9999 };                                                 // in [m]
//    std::vector<double> CoG_fuel{ 9999 };                                                     // in [m]
//    std::vector<double> angle_of_attack{ 0.0 };                                               // in [deg]
//    std::vector<double> glidepath_angle{ 0.0 };                                               // in [deg]
//    std::vector<double> stabilizer_incidence_angle{ 0.0 };                                    // in [deg]
//
//    // fill the containers with data from the mission analysis
//    for (auto seg : this->segments)
//    {
//        total_time.push_back(total_time.back() + seg.delta_t);
//        distance_traveled.push_back(distance_traveled.back() + seg.range);
//        altitude.push_back(seg.altitude);
//        flight_level.push_back(seg.altitude * meter_to_feet / 100);
//        std::string vertical_acceleration{ (seg.dv_dt > 0) ? "accelerated" :
//                                          (seg.dv_dt < 0) ? "decelerated" :
//                                           "constant" };
//        std::string horizantal_acceleration{ (seg.dh_dt > 0) ? "climb" :
//                                            (seg.dh_dt < 0) ? "descent" :
//                                            "cruise" };
//        mode_name.push_back(vertical_acceleration + "_" + horizantal_acceleration);
//        total_mass.push_back(seg.initial_weight / g0);
//        energy_carrier_ID.push_back(energy_carrier_ID.at(0));
//        thrust.push_back(seg.thrust);
//        shaftpower_offtake.push_back(0.0); // place holder
//        bleed_air.push_back(0.0); // place holder
//        double temp_fuelflow{ 0.0 };
//        std::vector<double> temp_N1{};
//        for (auto engine : this->engines)
//        {
//            std::string thrust_rating{};
//            switch (seg.type) {
//            case segment_type::Idle:
//                thrust_rating = "idle";
//                break;
//            case segment_type::Accelerated_Climb:
//                thrust_rating = "climb";
//                break;
//            case segment_type::Subsonic_Loiter:
//                thrust_rating = "cruise";
//                break;
//            case segment_type::Constant_Altitude_Speed_Cruise:
//                thrust_rating = "cruise";
//                break;
//            default:
//                break;
//            }
//            engine.calculate_N1_with_penalties(seg.altitude, seg.Mach, this->atm, 1.0, thrust_rating, bleed_air.back(), shaftpower_offtake.back());
//            temp_fuelflow += engine.get_fuelflow();
//            temp_N1.push_back(engine.get_operating_point().N);
//        }
//        fuelflow.push_back(temp_fuelflow);
//        fuel_consumed.push_back(total_mass.at(0) - total_mass.back());
//        energy_consumed.push_back(fuel_energy_density * fuel_consumed.back());
//        mach.push_back(seg.Mach);
//        cas.push_back(this->atm.getSpeedOfSound(seg.altitude) * seg.Mach); 
//        tas_in_meters_per_sec.push_back(this->atm.getSpeedOfSound(seg.altitude) * seg.Mach);
//        tas_in_knots.push_back(this->atm.getSpeedOfSound(seg.altitude) * seg.Mach * this->meter_per_sec_to_kts);
//        roc.push_back(seg.dh_dt * this->meter_to_feet * 60);
//        sar.push_back(tas_in_meters_per_sec.back() / fuelflow.back());
//        aero_config.push_back("clean");
//        CL.push_back(seg.CL);
//        spoiler_factor.push_back(0.); // TBD
//        double kinematic_viscosity{ this->atm.getViscosity(seg.altitude) / this->atm.getDensity(seg.altitude) };
//        raynolds_number.push_back(tas_in_meters_per_sec.back() * mean_aerodynamic_chord / kinematic_viscosity);
//        engine_rating.push_back(100.); // place holder
//        engine1_N1.push_back(temp_N1.at(0));
//        engine2_N1.push_back(temp_N1.at(1));
//        CoG_aircraft.push_back(25.); // place holder
//        CoG_fuel.push_back(25.); // place holder
//        angle_of_attack.push_back(std::asin(seg.dh_dt * tas_in_meters_per_sec.back()) * 180. / std::numbers::pi); // place holder
//        glidepath_angle.push_back(std::asin(seg.dh_dt * tas_in_meters_per_sec.back()) * 180. / std::numbers::pi);
//        stabilizer_incidence_angle.push_back(0.); // will be deleted
//    }
//};