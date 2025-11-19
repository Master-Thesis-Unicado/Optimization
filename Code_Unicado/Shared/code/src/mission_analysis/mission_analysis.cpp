/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

//#ifndef CONSTRAINTANALYSIS_CONSTRAINTANALYSIS_H_
//#define CONSTRAINTANALYSIS_CONSTRAINTANALYSIS_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>
#include <atmosphere/atmosphere.h>
#include <aixml/node.h>
#include "mission_analysis/mission_analysis.h"

namespace Mattingly
{
    namespace mission_analysis
    {
        atmosphere atm;
        double k_TO = 1.1;
        double g0 = 9.81;
        double a_standard = atm.getSpeedOfSound(0.0);
        double friction_coeff = 0.05;

        void constant_speed_climb(Segment segment) {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC / (segment.Mach * sqrt(nondim_temperature) * a_standard) * 
                (segment.dh_dt*segment.delta_t / 
                    (1 - (segment.CD / segment.CL) * (segment.initial_weight / segment.thrust))));
        };

        void horizontal_acceleration(Segment segment) {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC / (segment.Mach * sqrt(nondim_temperature) * a_standard) *
                ((pow(segment.Mach * atm.getSpeedOfSound(segment.altitude) + segment.dv_dt*segment.delta_t, 2) - pow(segment.Mach * atm.getSpeedOfSound(segment.altitude), 2))/(2*g0)) /
                    (1 - (segment.CD / segment.CL) * (segment.initial_weight / segment.thrust)));
        };

        void accelerated_climb(Segment segment) {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC / (segment.Mach * sqrt(nondim_temperature) * a_standard) *
                (segment.dh_dt * segment.delta_t + (pow(segment.Mach * atm.getSpeedOfSound(segment.altitude) + segment.dv_dt * segment.delta_t, 2) - pow(segment.Mach * atm.getSpeedOfSound(segment.altitude), 2)) / (2 * g0)) /
                (1 - (segment.CD / segment.CL) * (segment.initial_weight / segment.thrust)));
        };

        void takeoff_acceleration(Segment segment) {
            double q = 0.5 * atm.getDensity(segment.altitude) * pow(atm.getSpeedOfSound(segment.altitude) * segment.Mach, 2);
            double u = (segment.CD * (q / segment.initial_weight_fraction) * (1 / segment.wing_loading) + friction_coeff) * (segment.initial_weight_fraction / segment.thrust_lapse) * (1 / segment.thrust_to_weight);
            double V_TO = k_TO * 1; /*@todo get the stall speed from somewhere*/

            segment.weight_fraction = exp(
                -segment.TSFC / g0 *
                (V_TO / (1 - u))); 
        };

        void constant_altitude_cruise(Segment segment) {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);

            segment.weight_fraction = exp(
                -segment.TSFC / (segment.Mach * sqrt(nondim_temperature) * a_standard) *
                (segment.CD / segment.CL) *
                segment.range);
        };

        void constant_altitude_turn(Segment segment) {
            double nondim_temperature = atm.getTemperature(segment.altitude) / atm.getTemperature(0.0);
            double q = 0.5 * atm.getDensity(segment.altitude) * pow(atm.getSpeedOfSound(segment.altitude) * segment.Mach, 2);
            double CL = segment.load_factor * segment.initial_weight_fraction / q * segment.wing_loading;

            segment.weight_fraction = exp(
                -segment.TSFC *
                (segment.CD / (CL / segment.load_factor)) *
                (segment.Mach * atm.getSpeedOfSound(segment.altitude) * 2 * std::numbers::pi) /
                (g0 * sqrt(pow(segment.load_factor, 2) - 1)));
        };

        void BSCM(Segment segment) {}; /*@todo finalize this*/

        void subsonic_loiter(Segment segment) {
            double EF = segment.CL / (segment.CD * segment.TSFC);

            segment.weight_fraction = exp(-segment.delta_h / EF);
        };

        void warm_up(Segment segment) {
            segment.weight_fraction = 1 - segment.TSFC * segment.thrust_lapse / segment.initial_weight_fraction * segment.thrust_lapse * segment.delta_h;
        };

        void takeoff_rotation(Segment segment) {
            segment.weight_fraction = 1 - segment.TSFC * segment.thrust_lapse / segment.initial_weight_fraction * segment.thrust_lapse * segment.delta_h;
        };

        void constant_energy_height_maneuver(Segment segment) {
            segment.weight_fraction = exp(-segment.TSFC * segment.CD / segment.CL * segment.delta_h);
        };

        void idle_thrust(Segment segment) {
            segment.weight_fraction = segment.TSFC * segment.thrust * segment.delta_h;
        };
    };
};