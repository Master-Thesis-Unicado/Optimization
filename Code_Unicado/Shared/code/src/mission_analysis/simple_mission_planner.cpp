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
#include <ctime>
#include "../../include/mission_analysis/simple_mission_analysis.h"
#include "../../include/constraint_analysis/ca_parser.h"

void Mission::plan_simple_mission(const std::shared_ptr<node>& mission_node)
{
    // initialize the mission definition elements
    auto mission_definition = mission_node->getChildren();
    Check_Point initial_step;
    Check_Point next_step;
    double throttle_setting = 85.;
    clock_t start_time;

    // read the start node
    const Check_Point start_node(
        mission_node->at("start/altitude"),
        mission_node->at("start/Mach"),
        0.0,
        mission_node->at("start/weight_fraction"));
    initial_step = start_node;

    // initialize the polar
    readPolar polar(this->polar_xml, "clean", "linear", initial_step.M);

    // delete the start node from the vector of mission steps
    mission_definition.erase(mission_definition.begin());

    // while end node is not reached
    while ((*mission_definition.begin())->name != "end")
    {
        start_time = clock();
        // read next node
        auto mission_step = *mission_definition.begin();

        // if the next node is a phase
        if (mission_step->name == "phase")
        {
            int index = 0;

            Check_Point phase_start = Check_Point(
                initial_step.h,
                initial_step.M,
                initial_step.distance_covered,
                initial_step.weight_fraction);

            std::string step_type = mission_step->at("/type");
            // if the phase type is cruise
            if (step_type == "cruise")
            {
                // get cruise conditions
                const double cruise_altitude = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/altitude");
                const double cruise_M = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/Mach");

                // get cruise end conditions
                const double total_range = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/range") * 1000.0;
                const double final_altitude = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/end/altitude");
                const double final_M = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/end/Mach");

                // initialize where the segment will start moving towards the segment end conditions
                double  segment_transition = total_range * 0.5;

                // set target cruise checkpoint
                Check_Point cruise_step = Check_Point(
                    cruise_altitude,
                    cruise_M,
                    initial_step.distance_covered + total_range - segment_transition,
                    initial_step.weight_fraction);

                Check_Point end_step = Check_Point(
                    final_altitude,
                    final_M,
                    initial_step.distance_covered + total_range,
                    initial_step.weight_fraction);

                if (Mission::delta_energy_height(cruise_step, end_step) < -0.1)
                {
                    segment_transition = Mission::estimate_transition_point(cruise_step, end_step, polar);
                    cruise_step.distance_covered = initial_step.distance_covered + total_range - segment_transition;
                }

                // create a vector of checkpoings
                std::vector<Check_Point> transition_points = { cruise_step, end_step };

                // while loop for going to the target
                bool valid_segment = false;

                // reset the throttle setting
                throttle_setting = 85.;

                while (!valid_segment) {
                 
                    // for each step in cruise checkpoints move towards the checkpoint
                    for (auto target_step : transition_points) {

                        // while the new conditions are not matching the target
                        while ((target_step.h - next_step.h)/ target_step.h > 10. ||
                            (target_step.M - next_step.M)/target_step.M > 0.01 ||
                            next_step.distance_covered < target_step.distance_covered)
                        {

                            // update the polar if Mach number is different
                            if (polar.allowable_Mach.size() > 1)
                            {
                                auto it = std::min_element(polar.allowable_Mach.begin(), polar.allowable_Mach.end(),
                                    [initial_step](double a, double b) {
                                        return std::abs(a - initial_step.M) < std::abs(b - initial_step.M);
                                    });
                                
                                int new_idx = std::distance(polar.allowable_Mach.begin(), it);
                                // If the index of the closes element is different, read that element
                                if (index != new_idx)
                                {
                                    index = new_idx;
                                    polar.read_polar(polar.allowable_Mach[index]);
                                }
                            };

                            if (Mission::delta_energy_height(initial_step, target_step) < -0.1)
                            {
                                // go to the next checkpoint with throttle setting = 0.
                                try {
                                    next_step = Mission::to_next_checkpoint(initial_step, target_step, 0., this->thrust_share, polar);
                                }
                                catch (const std::exception& e) {
                                    std::cout << "Descent attempt failed: taking a step back \n";
                                    break;
                                }
                                
                                double next_Es = Mission::delta_energy_height(initial_step, next_step);
                                if (next_Es > 0.001) {
                                    break;
                                };
                            }
                            else if(Mission::delta_energy_height(initial_step, target_step) > 0.1)
                            {
                                // go to the next checkpoint
                                next_step = Mission::to_next_checkpoint(initial_step, target_step, throttle_setting, this->thrust_share, polar);

                                if (std::abs((next_step.M - initial_step.M) / initial_step.M) < 0.001 && throttle_setting < 99. && target_step.M - next_step.M > 0.001)
                                {
                                    throttle_setting += 1.;
                                }
                            }
                            else
                            {
                                next_step = Mission::to_next_checkpoint(initial_step, target_step, 85., this->thrust_share, polar);
                            }
                            // move to the next checkpoint
                            initial_step = next_step;
                        };
                    };

                    // valid descend when distance covered in total is accurate enough (less than 1% deviation)
                    valid_segment = (next_step.distance_covered - end_step.distance_covered) / end_step.distance_covered < 0.01;
                    valid_segment = valid_segment && std::abs((next_step.M - end_step.M) / end_step.M) < 0.01;
                    valid_segment = valid_segment && std::abs((next_step.h - end_step.h) / end_step.h) < 0.01;

                    if (!valid_segment) {
                        if (Mission::delta_energy_height(phase_start, end_step) < 0.)
                        {
                            std::cout << "Descent attempt failed: taking a step back \n";

                            // set new checkpoints
                            cruise_step.distance_covered = cruise_step.distance_covered - segment_transition * 0.1;

                            while (this->segments.back().range > cruise_step.distance_covered)
                            {
                                this->segments.pop_back();
                            };

                            transition_points = { end_step };

                            // go back to the phase start
                            initial_step = Check_Point(
                                this->segments.back().altitude,
                                this->segments.back().Mach,
                                this->segments.back().range,
                                this->segments.back().initial_weight / (this->estimated_WTO*this->g0));;

                            next_step = initial_step;
                        }
                        else
                        {
                            std::cout << "Climb attempt failed: increasing default throttle \n";

                            // set new checkpoints
                            cruise_step.distance_covered = cruise_step.distance_covered - segment_transition * 0.05;

                            while (this->segments.back().range > cruise_step.distance_covered)
                            {
                                this->segments.pop_back();
                            };

                            throttle_setting += 2.5;

                            transition_points = { end_step };

                            // go back to the phase start
                            initial_step = Check_Point(
                                this->segments.back().altitude,
                                this->segments.back().Mach,
                                this->segments.back().range,
                                this->segments.back().initial_weight / (this->estimated_WTO * this->g0));;

                            next_step = initial_step;
                        };                        
                    };
                };

                // the target checkpoint is reached - go out of the loop here
                mission_definition.erase(mission_definition.begin());

                double duration = double(clock() - start_time) / double(CLOCKS_PER_SEC);

                // report
                std::string tag = mission_node->at("phase@" + mission_step->getStringAttrib("ID") + "/tag");
                std::cout << "step " << tag << " calculated successfully in " << std::fixed << duration << std::setprecision(3); 
                std::cout << " sec \n";
            };
        };
    };
    // exit if the next node is the end node
}

auto Mission::to_next_checkpoint(const Check_Point &initial_step, const Check_Point &target_step, double throttle_setting, const std::vector<double>& thrust_share, readPolar& polar)  -> Check_Point
{
    // initialize the segment
    Segment current_segment;

    // get the altitude difference - in the loop
    const double h_difference = target_step.h - initial_step.h;

    // get the velocity difference - in the loop
    const double v_difference = target_step.M * this->atm.getSpeedOfSound(target_step.h) - initial_step.M * this->atm.getSpeedOfSound(initial_step.h);

    // get Thrust and TSFC - put this outside of the loop as an input
    double T = 0.0;
    double TSFC = 0.0;
    int i = 0;

    for (auto engine : this->engines)
    {
        if (throttle_setting > 100.)
        {
            double T_max_minus_10 = engine.get_thrust_with_lever_position(0.9, initial_step.M, initial_step.h);
            double T_max = engine.get_thrust_with_lever_position(1., initial_step.M, initial_step.h);
            T += T_max + (T_max - T_max_minus_10) / 10. * (throttle_setting - 100.);
        }
        else
        {
            T += engine.get_thrust_with_lever_position(throttle_setting / 100., initial_step.M, initial_step.h);
        }

        TSFC += engine.get_tsfc() * thrust_share[i];
        i++;
    };

    // get weight fraction and weight - in the loop
    double W = initial_step.weight_fraction * this->estimated_WTO * this->g0;

    // get CL and CD - in the loop
    double CL = polar.getCL(this->estimated_WTO * this->g0 / this->S_Ref, 1.0, initial_step.weight_fraction, initial_step.M, initial_step.h);

    double CD = polar.interpCD(CL);
    // double CD = avl_interface::getCD(initial_step.M, CL);

    // calculate the initial velocity - in the loop
    double v = initial_step.M * this->atm.getSpeedOfSound(initial_step.h);

    // time delta 60 seconds delta by default - in the loop
    double time_delta = 60.;

    // calculate the drag
    double drag = 0.5 * atm.getDensity(initial_step.h) * pow(v, 2.) * CD * this->S_Ref;

    double dh_dt = 0.;
    double dv_dt = 0.;

    // if dh/dt and dv/dt are both positive divide the SEP to both 50%
    if (h_difference > 0.001 && v_difference > 0.001) {
        dh_dt =
            (T - drag) * v /
            (2. * W);

        if (dh_dt / v > std::sin(std::numbers::pi / 4.))
        {
            dh_dt = v * std::sin(std::numbers::pi / 4.);
        }
        dv_dt =
            ((T - drag) * v / W - dh_dt) * this->g0 / v;
    }
    // if dh/dt and dv/dt are both negative follow constant descent gradient
    else if (h_difference < -0.001 && v_difference < -0.001) {
        time_delta = 10.;
        double del_x = target_step.distance_covered - initial_step.distance_covered;
        double del_h = target_step.h - initial_step.h;
        double v_avg = (target_step.M * this->atm.getSpeedOfSound(target_step.h) + initial_step.M * this->atm.getSpeedOfSound(initial_step.h))/2.;
        dh_dt = del_h / del_x * v_avg;
        dv_dt = ((T - drag) / W * v - dh_dt) * this->g0 / v;
    }
    else if (h_difference < -0.001) {
        time_delta = 1.;
        dh_dt = v * ((T - drag)/W);
    }
    else if (v_difference < -0.001) {
        time_delta = 1.;
        dv_dt = ((T - drag) / W) * this->g0;
    }
    // if there is only one, give full SEP
    else if (h_difference > 0.001) {
        dh_dt = (T - drag) * v / W;
    }
    else if (v_difference > 0.001) {
        dv_dt = (T - drag) * this->g0 / (2. * W);
    };

    // calculate the time delta based on velocity difference and altitude difference - in the loop
    if ((target_step.distance_covered - initial_step.distance_covered) / v < time_delta && target_step.distance_covered>initial_step.distance_covered)
    {
        time_delta = (target_step.distance_covered - initial_step.distance_covered) / v + 0.01;
    };

    if (std::sqrt(pow(dh_dt,2.)) > 0.001 || std::sqrt(pow(dv_dt, 2.)) > 0.001)
    {
        // decrease the time delta to 10 seconds if there is climb or acceleration
        double time_delta_1 = 10.;
        double time_delta_2 = 10.;
        // if the velocity or altitude can be reached, further decrease the time delta
        if (pow(dh_dt * time_delta_1, 2.) - pow(h_difference, 2.) > 0.001) {
            time_delta_1 = h_difference / dh_dt;
        };
        if (pow(dv_dt * time_delta_2,2.) - pow(v_difference, 2.) > 0.001) {
            time_delta_2 = v_difference / dv_dt;
        };
        time_delta = std::min(time_delta_1, time_delta_2);
    };

    if (dh_dt < -0.001 || dv_dt < -0.001)
    {
        current_segment = Segment(
            segment_type::Idle,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            initial_step.distance_covered,
            1.0, // load factor
            CD,
            CL,
            TSFC,
            T);
        current_segment = Mattingly::mission_analysis::idle_thrust(current_segment, W);
    }
    else if (dh_dt > 0.001 || dv_dt > 0.001)
    {
        // if dh/dt and dv/dt are both positive - in the loop
        current_segment = Segment(
            segment_type::Accelerated_Climb,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            initial_step.distance_covered,
            1.0, // load factor
            CD,
            CL,
            TSFC,
            T);
        current_segment = Mattingly::mission_analysis::evaluate_segment(current_segment);
    }
    else if (dh_dt == 0.0 && dv_dt == 0.0)
    {
        current_segment = Segment(
            segment_type::Constant_Altitude_Speed_Cruise,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            initial_step.distance_covered,
            1.0, // load factor
            CD,
            CL,
            TSFC,
            T);
        current_segment = Mattingly::mission_analysis::evaluate_segment(current_segment);
    }
    else
    {
        current_segment = Segment(
            segment_type::Constant_Altitude_Speed_Cruise,
            W,
            dh_dt,
            dv_dt,
            time_delta,
            initial_step.h,
            initial_step.M,
            initial_step.distance_covered,
            1.0, // load factor
            CD,
            CL,
            TSFC,
            T);
        current_segment = Mattingly::mission_analysis::evaluate_segment(current_segment);
    }

    // get new conditions based on time interval - in the loop
    double next_h = initial_step.h + dh_dt * time_delta;
    double next_M = (v + dv_dt * time_delta) / this->atm.getSpeedOfSound(next_h);
    double next_weight_fraction = current_segment.weight_fraction * initial_step.weight_fraction;
    double FPA = std::asin(dh_dt / v);
    double distance_covered = v * std::cos(FPA) * time_delta + 0.5 * dv_dt * pow(time_delta, 2) * std::cos(FPA) + initial_step.distance_covered;
    Check_Point next_step(next_h, next_M, distance_covered, next_weight_fraction);
    this->segments.push_back(current_segment);
    return next_step;
};

auto Mission::delta_energy_height(const Check_Point& initial_step, const Check_Point& target_step) -> double
{
    // calculate the initial energy height
    double initial_energy_height = initial_step.h + pow(initial_step.M * this->atm.getSpeedOfSound(initial_step.h), 2) / (2. * 9.81);

    // calculate the final energy height
    double final_energy_height = target_step.h + pow(target_step.M * this->atm.getSpeedOfSound(target_step.h), 2) / (2. * 9.81);

    return final_energy_height - initial_energy_height;
}

auto Mission::estimate_transition_point(const Check_Point &cruise_step, const Check_Point &final_step, readPolar& polar) -> double
{
    double del_h = final_step.h - cruise_step.h;
    double v_final = final_step.M * this->atm.getSpeedOfSound(final_step.h);
    double v_initial = cruise_step.M * this->atm.getSpeedOfSound(cruise_step.h);
    double del_v = v_final - v_initial;
    double weight = cruise_step.weight_fraction * this->estimated_WTO * this->g0;
    double T = 0.;
    for (auto engine : this->engines)
    {
        T += engine.get_thrust_with_lever_position(0. / 100., cruise_step.M, cruise_step.h);
    };
    double CL = polar.getCL(this->estimated_WTO * this->g0 / this->S_Ref, 1.0, cruise_step.weight_fraction, cruise_step.M, cruise_step.h);
    double CD = polar.interpCD(CL);
    double drag = 0.5 * atm.getDensity(cruise_step.h) * pow(v_initial, 2.) * CD * this->S_Ref;
    double transition_del_t = (del_h + v_initial * del_v / this->g0) * weight / ((T - drag) * v_initial);

    return (v_initial * transition_del_t + 0.5 * (v_final - v_initial) * transition_del_t);
}