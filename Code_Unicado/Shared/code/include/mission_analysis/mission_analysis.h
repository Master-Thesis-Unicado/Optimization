/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef CONSTRAINTANALYSIS_MISSIONANALYSIS_H_
#define CONSTRAINTANALYSIS_MISSIONANALYSIS_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>

enum class segment_type : int
{
    Idle = 0,
    Constant_Speed_Climb = 1,
    Horizontal_Acceleration = 2,
    Accelerated_Climb = 3,
    Takeoff_Acceleration = 4,
    Constant_Altitude_Cruise = 5,
    Constant_Altitude_Turn = 6,
    Best_Subsonic_Cruise_M_and_h = 7,
    Subsonic_Loiter = 8,
    Warm_Up = 9,
    Takeoff_Rotation = 10,
    Constant_Energy_Height_Maneuver = 11
};

// accelerated climb to reach KCAS, climb to TOC, cruise, descent climbs--> 85% throttle 
struct Segment
{
    segment_type type = segment_type::Idle;
    double weight_fraction = 1.0;                           /* Weight fraction of the whole segment */
    double initial_weight_fraction = 1.0;                   /* Input: Initial weight fraction at the start of the segment */
    double initial_weight = 1.0                             /* Input: Initial weight at the start of the segment */
    double delta_h = 0.0;                                   /* Input: Difference between the final and initial altitude */
    double delta_v = 0.0;                                   /* Input: Difference between the final and initial speed */
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
    {
        evaluate_segment()
    };

    Segment(
        segment_type type
    ) :
        type(type)
    {
        evaluate_segment()
    };

    void evaluate_segment()
    {
        switch (this->type)
        {
        case segment_type::Constant_Speed_Climb:
            Mattingly::mission_analysis::constant_speed_climb(*this);
            break;
        case segment_type::Horizontal_Acceleration:
            Mattingly::mission_analysis::horizontal_acceleration(*this);
            break;
        case segment_type::Accelerated_Climb:
            Mattingly::mission_analysis::accelerated_climb(*this);
            break;
        case segment_type::Takeoff_Acceleration:
            Mattingly::mission_analysis::takeoff_acceleration(*this);
            break;
        case segment_type::Constant_Altitude_Cruise:
            Mattingly::mission_analysis::constant_altitude_cruise(*this);
            break;
        case segment_type::Constant_Altitude_Turn:
            Mattingly::mission_analysis::constant_altitude_turn(*this);
            break;
        case segment_type::Best_Subsonic_Cruise_M_and_h:
            Mattingly::mission_analysis::BSCM(*this);
            break;
        case segment_type::Subsonic_Loiter:
            Mattingly::mission_analysis::subsonic_loiter(*this);
            break;
        case segment_type::Warm_Up:
            Mattingly::mission_analysis::warm_up(*this);
            break;
        case segment_type::Takeoff_Rotation:
            Mattingly::mission_analysis::takeoff_rotation(*this);
            break;
        case segment_type::Constant_Energy_Height_Maneuver:
            Mattingly::mission_analysis::constant_energy_height_maneuver(*this);
            break;
        case segment_type::Idle:
            Mattingly::mission_analysis::idle_thrust(*this);
            break;
        default:
            break;
        }
    };
};

struct Check_Point
{
    double h = 0.0;
    double M = 0.0;
    double weight_fraction = 1.0;
    double distance_covered = 0.0;
    

    Check_Point() = default;

    Check_Point(double h, double M, double distance_covered, double weight_fraction) :
        h(h), M(M), distance_covered(distance_covered), weight_fraction(weight_fraction){};

};

namespace Mattingly
{
    namespace mission_analysis
    {
        void constant_speed_climb(Segment segment);

        void horizontal_acceleration(Segment segment);

        void accelerated_climb(Segment segment);

        void takeoff_acceleration(Segment segment);

        void constant_altitude_cruise(Segment segment);

        void constant_altitude_turn(Segment segment);

        void BSCM(Segment segment);

        void subsonic_loiter(Segment segment);

        void warm_up(Segment segment);

        void takeoff_rotation(Segment segment);

        void constant_energy_height_maneuver(Segment segment);

        void idle_thrust(Segment segment);
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
    double estimated_WTO = 0.0;
    double mission_weight_fraction = 1.0;
    const std::shared_ptr<node>& polar_xml;
    const std::shared_ptr<node>& mission_xml;
    const Engine engine;
    atmosphere atm;
    const double S_Ref = 1.0;
    const double g0 = 9.81;

    Mission(std::shared_ptr<node>& polar_xml, std::shared_ptr<node>& mission_xml, Engine engine) : polar_xml(polar_xml), mission_xml(mission_xml), engine(engine)
    { 
        calculate_WTO();
    };

    void calculate_WTO() {
        for (auto segment : this->segments)
        {
            this->mission_weight_fraction *= segment.weight_fraction;
        }
        this->mission_WTO = this->mission_WP / (this->mission_weight_fraction - this->mission_WE / this->estimated_WTO);
        this->mission_Wf = this->mission_WTO * (1 - this->mission_weight_fraction);
    };

    void plan_mission(const std::shared_ptr<node>& mission_node);

    auto to_next_checkpoint(Check_Point initial_step, Check_Point target_step, double throttle_setting) -> Check_Point;
};

#endif