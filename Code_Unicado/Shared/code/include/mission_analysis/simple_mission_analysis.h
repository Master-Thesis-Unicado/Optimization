/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef MISSIONANALYSIS_MISSIONANALYSIS_H_
#define MISSIONANALYSIS_MISSIONANALYSIS_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>
#include <engine/engine.h>
#include "../constraint_analysis/ca_parser.h"

enum class segment_type : int
{
    Idle = 0,
    Accelerated_Climb = 1,
    Subsonic_Loiter = 2,
    Constant_Altitude_Speed_Cruise = 3
};

struct Segment
{
    segment_type type = segment_type::Idle;
    double weight_fraction = 1.0;                           /* Weight fraction of the whole segment */
    double initial_weight_fraction = 1.0;                   /* Input: Initial weight fraction at the start of the segment */
    double initial_weight = 1.0;                            /* Input: Initial weight at the start of the segment */
    double dh_dt = 0.0;                                     /* Input: Difference between the final and initial altitude */
    double dv_dt = 0.0;                                     /* Input: Difference between the final and initial speed */
    double delta_t = 60.;                                   /* Input: Duration of the segmnent */
    double altitude = 0.0;                                  /* Input: Initial altitude at the start of the segment */
    double Mach = 0.0;                                      /* Input: Mach number throughout the segment */
    double range = 0.0;                                     /* Input: Range of the segment */
    double throttle = 100.0;                                /* Input: Throttle setting throughout the segment */
    double load_factor = 1.0;                               /* Input: Initial weight fraction at the start of the segment */
    double CD = 0.01;                                       /* Polar: Drag Coefficient at segment conditions */
    double CL = 0.02;                                       /* Polar: Lift Coefficient at segment conditions */
    double TSFC = 0.5;                                      /* EnLib: Thrust Specific Fuel Consumption of the engine */
    double thrust_lapse = 1.0;                              /* EnLib: Thrust lapse of the engine at altitude and mach number */
    double thrust = 1.2;                                    /* AcXML: Thrust to weight ratio of the aircraft */
    double wing_loading = 1000;                             /* AcXML: Wing loading of the aircraft */

    Segment() = default;

    Segment(
        segment_type type,
        double initial_weight, 
        double dh_dt,
        double dv_dt,
        double delta_t,
        double altitude,
        double Mach,
        double range, 
        double load_factor,
        double CD,
        double CL,
        double TSFC,
        double thrust
    ) :
        type(type), 
        initial_weight(initial_weight), 
        dh_dt(dh_dt),
        dv_dt(dv_dt),
        delta_t(delta_t),
        altitude(altitude),
        Mach(Mach),
        range(range), 
        load_factor(load_factor),
        CD(CD),
        CL(CL),
        TSFC(TSFC),
        thrust(thrust) 
    {    };

    Segment(
        segment_type type
    ) :
        type(type)
    {    };
};

struct Check_Point
{
    double h = 0.0;
    double M = 0.0;
    double distance_covered = 0.0;
    double weight_fraction = 1.0;    

    Check_Point() = default;

    Check_Point(double h, double M, double distance_covered, double weight_fraction) :
        h(h), M(M), distance_covered(distance_covered), weight_fraction(weight_fraction){};

};

namespace Mattingly
{
    namespace mission_analysis
    {
        auto accelerated_climb(Segment segment) -> Segment;

        auto subsonic_loiter(Segment segment) -> Segment;

        auto idle_thrust(Segment segment, double W) -> Segment;

        auto constant_altitude_cruise(Segment segment) -> Segment;

        auto evaluate_segment(Segment segment) -> const Segment;
    };
};

class Mission
{
public:
    std::vector<Segment> segments = {};
    double mission_WTO = 0.0;
    double mission_Wf = 0.0;
    double mission_WE = 0.0;
    double mission_WP = 0.0;
    double estimated_WTO = 75000.0;
    double mission_weight_fraction = 1.0;
    const std::shared_ptr<node>& polar_xml;
    const std::shared_ptr<node>& mission_xml;
    const std::vector<Engine> engines;
    const std::vector<double> thrust_share = {};
    atmosphere atm;
    const double S_Ref = 150.0;
    const double g0 = 9.81;

    Mission(const std::shared_ptr<node>& polar_xml, const std::shared_ptr<node>& mission_xml, const std::vector<Engine> engines, const std::vector<double> thrust_share) : 
        polar_xml(polar_xml), mission_xml(mission_xml), engines(engines), thrust_share(thrust_share)
    { 
        plan_simple_mission(mission_xml);
        calculate_WTO();
        visualize_mission();
        // create_csv_out();
    };

    void calculate_WTO() {
        for (auto segment : this->segments)
        {
            this->mission_weight_fraction *= segment.weight_fraction;
        }
        this->mission_WTO = this->mission_WP / (this->mission_weight_fraction - this->mission_WE / this->estimated_WTO);
        this->mission_Wf = this->mission_WTO * (1. - this->mission_weight_fraction);
    };

    void plan_simple_mission(const std::shared_ptr<node>& mission_node);

    auto to_next_checkpoint(const Check_Point& initial_step, const Check_Point& target_step, double throttle_setting, const std::vector<double> &thrust_share, readPolar& polar) -> Check_Point;

    auto delta_energy_height(const Check_Point& initial_step, const Check_Point& target_step) -> double;

    auto estimate_transition_point(const Check_Point &cruise_step, const Check_Point& final_step, readPolar& polar) -> double;

    void visualize_mission();

    // void create_csv_out();
};

#endif