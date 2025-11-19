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
#include "mission_analysis/mission_analysis.h"
#include "constraint_analysis/ca_parser.h"

Mission::plan_mission(const std::shared_ptr<node>& mission_node)
{
    // initialize the mission definition elements
    auto mission_definition = mission_node.getChildren();
    Check_Point initial_step;
    Check_Point next_step;
    Check_Point target_step;
    double throttle_setting = 0.;

    // read the start node
    const Check_Point start_node(
        mission_node->at("start/altitude"),
        mission_node->at("start/Mach"),
        mission_node->at("start/weight_fraction"),
        0.0);
    initial_step = start_node;

    // delete the start node from the vector of mission steps
    mission_definition.erase(mission_definition.begin());

    // while end node is not reached
    while (*mission_definition.begin()->name != "end")
    {
        // read next node
        auto mission_step = *mission_definition.begin();

        // if the next node is a checkpoint do
        if (mission_step->name == "checkpoint")
        {
            // set target step - outside of the loop
            target_step = Check_Point(
                mission_node->at("checkpoint@" + mission_step->getStringAttrib("ID") + "/altitude"),
                mission_node->at("checkpoint@" + mission_step->getStringAttrib("ID") + "/Mach"),
                mission_node->at("checkpoint@" + mission_step->getStringAttrib("ID") + "/distance"),
                initial_step.weight_fraction);

            // calculate the energy height difference
            if (Mission::delta_energy_height(initial_step, target_step) > 1.) {
                // if the enrgy height difference is positive set initial throttle to 85%
                throttle_setting = 85.;
            }
            else if (Mission::delta_energy_height(initial_step, target_step) < -1.)
            {
                // if the enrgy height difference is negative set initial throttle to 0%
                throttle_setting = 0.;
            }
            else
            {
                // if the enrgy height difference is zero, set the throttle to 0% but it will be recalculated during the segment
                throttle_setting = 0.
            };

            // if the distance covered in target step is set to 0, it is not taken into consideration
            while (target_step.distance_covered != 0 && (next_step.distance_covered - target_step.distance_covered) / (target_step.distance_covered + 0.01) > 0.05)
            {
                // while the new conditions are not matching the target
                while (next_step.h != target_step.h || next_step.M != target_step.M || next_step.distance_covered < target_step.distance_covered)
                {
                    // go to the next checkpoint
                    next_step = Mission::to_next_checkpoint(initial_step, target_step, throttle_setting);

                    // move to the next checkpoint
                    initial_step = next_step;
                };

                // increase the throttle setting for the next loop
                // the next loop gets executed if there is an overshoot, meaning that the aircraft cannot accelerate or climb within the given distance
                // therefore the throttle setting needs to get increase
                throttle_setting += 2.5;
            };
            // the target checkpoint is reached
            mission_definition.erase(mission_definition.begin());
        };

        // if the next node is a phase
        if (mission_step->name == "phase")
        {
            auto phase_target = mission_step->findVector("/target", true, 1);

            // if the phase doesn't have a target
            if (phase_target.empty())
            {
                // if the phase type is cruise
                if (mission_step->at("type/") == "cruise")
                {
                    // set target checkpoint by calculating the final position and altitude
                    target_step = Check_Point(
                        mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/altitude"),
                        mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/Mach"),
                        initial_step.distance_covered + mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/range")*1000.0,
                        initial_step.weight_fraction);

                    // while the new conditions are not matching the target
                    while (next_step.h != target_step.h || next_step.M != target_step.M || next_step.distance_covered < target_step.distance_covered)
                    {
                        // go to the next checkpoint
                        next_step = Mission::to_next_checkpoint(initial_step, target_step);
                        // move to the next checkpoint
                        initial_step = next_step;
                    };
                    // the target checkpoint is reached - go out of the loop here
                    mission_definition.erase(mission_definition.begin());
                };
            }
            // if the phase has a target
            else
            {
                // if the phase type is cruise
                if (mission_step->at("type/") == "cruise")
                {
                    // set target checkpoint by getting the final mach, altitude, and cruise range
                    target_step = Check_Point(
                        mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/target/altitude"),
                        mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/target/Mach"),
                        initial_step.distance_covered + mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/range")*1000.0,
                        initial_step.weight_fraction);

                    // set the transition point where the phase transition happens
                    double transition_point = 0.5;

                    // the segment operates in a constant rate of altitude change, where the speed gets altered accordingly
                    while ((target_step.M - next_step.M) / target_step.M > 0.05) {

                        // set intermediate target to be reached during cruise
                        // the order of events are: cruise first, then take necessary steps to reach the target
                        // ensures that the range of (cruise + necessary steps) = phase range specified
                        Check_Point intermediate_step = Check_Point(
                            mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/target/altitude"),
                            mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/target/Mach"),
                            (initial_step.distance_covered + mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/range") * 1000.0 * transition_point),
                            initial_step.weight_fraction);

                        // set new step within the phase
                        std::vector<Check_Point> phase_checkpoints = { intermediate_step , target_step };

                        for (auto target_i : phase_checkpoints) {
                            // while the new conditions are not matching the target
                            while (next_step.h != target_i.h || next_step.distance_covered < target_i.distance_covered)
                            {
                                // go to the next checkpoint
                                next_step = Mission::to_next_checkpoint(initial_step, target_i);
                                // move to the next checkpoint
                                initial_step = next_step;
                            };
                        };

                        // update the transition point based on the final M
                        if ((target_step.M - next_step.M) > 0)
                        {
                            transition_point += 0.01; // can be updated for a better guess
                        }
                        else
                        {
                            transition_point -= 0.01; // can be updated for a better guess
                        }
                    };
                    // the target checkpoint is reached - go out of the loop here
                    mission_definition.erase(mission_definition.begin());
                };
            };
        };
    };
        // exit if the next node is the end node
}

auto Mission::to_next_checkpoint(Check_Point initial_step, Check_Point target_step, double throttle_setting)
{
    // initialize the segment
    Segment current_segment;

    // get the altitude difference - in the loop
    double h_difference = target_step.h - initial_step.h;

    // get the velocity difference - in the loop
    double v_difference = target_step.M * this->atm.getSpeedOfSound(target.h) - initial_step.M * this->atm.getSpeedOfSound(initial_step.h);

    // get Thrust and TSFC - put this outside of the loop as an input
    double T = 1000.0;
    double TSFC = 0.1;

    // get weight fraction and weight - in the loop
    double W = initial_step.weight_fraction * this->estimated_WTO;

    // initialize the polar file reader - can be out of the loop if the M is updated - in the loop for now
    readPolar polar(this->polar_xml, "clean", "linear", initial_step.M);

    // get CL and CD - in the loop
    double CL = polar.getCL(W, 1.0, initial_step.M, initial_step.h);
    double CD = polar.interpCD(CL);

    // calculate the initial velocity - in the loop
    double v = initial_step.M * this->atm.getSpeedOfSound(initial_step.h);

    // time delta 60 seconds delta by default - in the loop
    double time_delta = 60.;

    // if there is dh/dt and dv/dt divide the SEP to both 50% - in the loop
    double dh_dt = 0.;
    double dv_dt = 0.;
    if (h_difference > 0. && v_difference > 0.) {
        dh_dt =
            (T - 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref) * v /
            (2. * W);

        dv_dt =
            (T - 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref) * this->g0 /
            (4. * W);
    }
    // if there is only one, give full SEP - in the loop
    else if (h_difference > 0.) {
        dh_dt = (T - 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref) * v / W;
    }
    else if (v_difference > 0.) {
        dv_dt = (T - 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref) * this->g0 / (2. * W);
    }
    else if (h_difference < 0. && v_difference < 0.) {
        time_delta = 10.;
        double drag = 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref;
        dv_dt = ((T - drag) * v * (target_step.distance_covered - initial_step.distance_covered) / W -
            (target_step.h - initial_step.h) * v) / ((target_step.distance_covered - initial_step.distance_covered) * v / g0 +
                (target_step.h - initial_step.h) / 2 * time_delta);
    }

    // calculate the time delta based on velocity difference and altitude difference - in the loop
    if ((target_step.distance_covered - initial_step.distance_covered) / v < time_delta && target_step.distance_covered>initial_step.distance_covered)
    {
        time_delta = (target_step.distance_covered - initial_step.distance_covered) / v;
    };

    if (dh_dt > 0. || dv_dt > 0.)
    {
        // decrease the time delta to 10 seconds if there is climb or acceleration
        double time_delta_1 = 10.;
        double time_delta_2 = 10.;
        // if the velocity or altitude can be reached, further decrease the time delta
        if (dh_dt * time_delta_1 > h_difference) {
            time_delta_1 = h_difference / dh_dt;
        };
        if (dv_dt * time_delta_2 > v_difference) {
            time_delta_2 = h_difference / dv_dt;
        };
        time_delta = min(time_delta_1, time_delta_2);
    };

    // if dh/dt and dv/dt are both positive - in the loop
    if (dh_dt > 0. || dv_dt > 0.) {
        current_segment(
            segment_type::Accelerated_Climb,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            0.0, // range is not relevant for accelerated climb
            1.0, // load factor
            CD,
            CL,
            TSFC,
            T);
        this->segments.push_back(current_segment);
    }
    // if no SEP required, cruise - in the loop can be deleted
    else if (dh_dt == 0. || dv_dt == 0.)
    {
        current_segment(
            segment_type::Constant_Altitude_Cruise,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            v * time_delta, // range
            1.0, // load factor 
            CD,
            CL,
            TSFC,
            T);
        this->segments.push_back(current_segment);
    }
    else if (dh_dt < 0 && dv_dt < 0)
    {
        current_segment(segment_type::Idle);
        this->segments.push_back(current_segment)
    }

    // get new conditions based on time interval - in the loop
    double next_h = initial_step.h + dh_dt * time_delta;
    double next_M = std::ceil((v + dv_dt * time_delta) / this->atm.getSpeedOfSound(next_h) * 100.) / 100.;
    double next_weight_fraction = current_segment.weight_fraction;
    double distance_covered = v * std::cos(polar.interpalpha(CL)) * time_delta + 0.5 * dv_dt * pow(time_delta, 2) * std::cos(polar.interpalpha(CL)) + initial_step.distance_covered;

    return Check_Point(next_h, next_M, distance_covered, next_weight_fraction);
};

auto Mission::delta_energy_height(Check_Point initial_step, Check_Point target_step)
{
    // calculate the initial energy height
    double initial_energy_height = initial_step.h + pow(initial_step.M * this->atm.getSpeedOfSound(initial_step.h), 2) / (2 * 9.81);

    // calculate the final energy height
    double final_energy_height = target_step.h + pow(target_step.M * this->atm.getSpeedOfSound(target_step.h), 2) / (2 * 9.81);

    return final_energy_height-initial_energy_height
}