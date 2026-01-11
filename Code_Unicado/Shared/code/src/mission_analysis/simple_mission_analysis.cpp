/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>
#include <atmosphere/atmosphere.h>
#include <aixml/node.h>
#include "../../include/mission_analysis/simple_mission_analysis.h"

namespace Mattingly
{
    namespace mission_analysis
    {
        atmosphere atm;
        double k_TO = 1.1;
        double g0 = 9.81;
        double a_standard = atm.getSpeedOfSound(0.0);
        double friction_coeff = 0.05;

        auto accelerated_climb(Segment segment) -> Segment
        {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC * g0 / (segment.Mach * sqrt(nondim_temperature) * a_standard) *
                (segment.dh_dt * segment.delta_t + (pow(segment.Mach * atm.getSpeedOfSound(segment.altitude) + segment.dv_dt * segment.delta_t, 2.) - pow(segment.Mach * atm.getSpeedOfSound(segment.altitude), 2.)) / (2. * g0)) /
                (1. - (segment.CD / segment.CL) * (segment.initial_weight / segment.thrust)));
            return segment;
        };

        auto subsonic_loiter(Segment segment) -> Segment
        {
            double EF = segment.CL / (segment.CD * segment.TSFC * g0);

            segment.weight_fraction = exp(-segment.delta_t / EF);

            return segment;
        };

        auto idle_thrust(Segment segment, double W) -> Segment
        {
            segment.weight_fraction = 1. - segment.TSFC * g0 * segment.thrust * segment.delta_t / W;

            return segment;
        };

        auto constant_altitude_cruise(Segment segment)  -> Segment
        {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC * g0 / (segment.Mach * sqrt(nondim_temperature) * a_standard) *
                (segment.CD / segment.CL) *
                segment.Mach * atm.getSpeedOfSound(segment.altitude)*segment.delta_t);
            return segment;
        };

        auto evaluate_segment(Segment segment) -> const Segment
        {
            switch (segment.type)
            {
            case segment_type::Accelerated_Climb:
                segment = Mattingly::mission_analysis::accelerated_climb(segment);
                break;
            case segment_type::Subsonic_Loiter:
                segment = Mattingly::mission_analysis::subsonic_loiter(segment);
                break;
            case segment_type::Idle:
                segment = Mattingly::mission_analysis::idle_thrust(segment, 1.);
                break;
            case segment_type::Constant_Altitude_Speed_Cruise:
                segment = Mattingly::mission_analysis::constant_altitude_cruise(segment);
                break;
            default:
                break;
            }
            return segment;
        };
    };
};