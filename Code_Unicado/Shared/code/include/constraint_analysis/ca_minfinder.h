/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef SRC_CONSTRAINT_ANALYSIS_MINFINDER_H_
#define SRC_CONSTRAINT_ANALYSIS_MINFINDER_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <iostream>
#include <aixml/endnode.h>
#include <format>

struct Constraint
{
    std::vector<double> wing_loading = {};
    std::vector<double> thrust_to_weight = {};
    std::string constraint_case = "unknown";

    Constraint(std::vector<double> wing_loading, std::vector<double> thrust_to_weight, std::string constraint_case) :
        wing_loading(wing_loading), thrust_to_weight(thrust_to_weight), constraint_case(constraint_case) {};
};

class Analysis
{
public:
    std::vector<Constraint> constraint_list = {};
    std::shared_ptr<node> acXML = {};

    Analysis(std::vector<Constraint> constraint_list, std::shared_ptr<node> acXML) : constraint_list(constraint_list), acXML(acXML) {};

    Analysis(std::vector<Constraint> constraint_list) : constraint_list(constraint_list) {};

};

class Simple_Analysis : Analysis
{
public:
    std::vector<double> dominant_thrust_to_weight = {};
    std::vector<double> wing_loading_boundaries = {};
    std::vector<double> wing_loading_vec = {};
    double design_thrust_to_weight = 1000.0;
    double design_wing_loading = 0.0;

    Simple_Analysis(std::vector<Constraint> constraint_list, std::vector<double> boundaries) : Analysis(constraint_list)
    {
        dominant_thrust_to_weight = constraint_list[0].thrust_to_weight;
        wing_loading_vec = constraint_list[0].wing_loading;
        wing_loading_boundaries = boundaries;
    };

    Simple_Analysis(std::vector<Constraint> constraint_list, std::shared_ptr<node> acXML, std::vector<double> boundaries) :
        Analysis(constraint_list, acXML)
    {
        dominant_thrust_to_weight = constraint_list[0].thrust_to_weight;
        wing_loading_vec = constraint_list[0].wing_loading;
        wing_loading_boundaries = boundaries;
    };

    void find_dominant_curve() {
        for (Constraint element : constraint_list) {
            int i = 0;
            for (double dom : dominant_thrust_to_weight) {
                dominant_thrust_to_weight[i] = std::max(dom, element.thrust_to_weight[i]);
                i++;
            }
        }
    };

    void find_design_point(const double& safety_factor)
    {
        std::sort(wing_loading_boundaries.begin(), wing_loading_boundaries.end());
        std::vector<double> xlims = {};
        for (auto x_limit : wing_loading_boundaries) {
            auto it = std::distance(wing_loading_vec.begin(), std::lower_bound(wing_loading_vec.begin(), wing_loading_vec.end(), x_limit));
            xlims.push_back(it);  
        }
        std::vector<double> limited_TW(dominant_thrust_to_weight.begin() + xlims[0], dominant_thrust_to_weight.begin() + xlims[1]);
        int i = xlims[0];
        for (double point : limited_TW)
        {
            if (point <= design_thrust_to_weight) 
            { 
                design_thrust_to_weight = point;
                design_wing_loading = wing_loading_vec[i];
            }
            i++;
        }
        design_thrust_to_weight = design_thrust_to_weight * (1.0 + safety_factor);
    };

    void find_converged_design_point(const double& previous_TW, const double& previous_WS, const double& safety_factor, const double& buffer_factor)
    {
        std::sort(wing_loading_boundaries.begin(), wing_loading_boundaries.end());
        std::vector<double> xlims = {};
        for (auto x_limit : wing_loading_boundaries) {
            auto it = std::distance(wing_loading_vec.begin(), std::lower_bound(wing_loading_vec.begin(), wing_loading_vec.end(), x_limit));
            xlims.push_back(it);
        }
        std::vector<double> limited_TW(dominant_thrust_to_weight.begin() + xlims[0], dominant_thrust_to_weight.begin() + xlims[1]);
        int i = xlims[0];
        for (double point : limited_TW)
        {
            if (point <= design_thrust_to_weight)
            {
                design_thrust_to_weight = point;
                design_wing_loading = wing_loading_vec[i];
            }
            i++;
        }
        design_thrust_to_weight = design_thrust_to_weight * (1.0 + safety_factor);

        double relative_change_TW = sqrt(pow(1 - previous_TW / design_thrust_to_weight, 2));
        if (relative_change_TW <= buffer_factor/100.0) { design_thrust_to_weight = previous_TW; };
        double relative_change_WS = sqrt(pow(1 - previous_WS / design_wing_loading, 2));
        if (relative_change_WS <= buffer_factor/100.0) { design_wing_loading = previous_WS; };
    };

    template<typename T>
    auto create_endnode(const T value, const std::string& unit, const std::string& node_path, const std::string& description) -> Endnode<T> {
        /* Create the node an set its properties */
        Endnode<T> new_node{ node_path, description };
        new_node.set_value(value);
        new_node.set_unit(unit);
        /* Return the new node */
        return new_node;
    };

    void update_design_point()
    {
        std::string subPath = "sizing_point";
        node& sizing_Node = (*acXML)[subPath];

        design_wing_loading = design_wing_loading / 9.81;

        Endnode<double> wingLoading = create_endnode(design_wing_loading, "kg/m^2", subPath + "/wing_loading", "Maximum takeoff mass (MTOM) divided by wing area (Sref)");
        wingLoading.update(sizing_Node);

        Endnode<double> thrustToweightRatio = create_endnode(design_thrust_to_weight, "1", subPath + "/thrust_to_weight", "Total thrust (kN) divided by maximum aircraft weight (kN)");
        thrustToweightRatio.update(sizing_Node);

        aixml::saveDocument(*acXML);
    };

    void write_design_point(const std::string& filename)
    {
        std::ifstream infile(filename);
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        if (infile.peek() == std::ifstream::traits_type::eof())
        {
            file << "Design Point Number" << "," << "\t" << "T/W"  << "," << "\t" << "W/S" << std::endl;
            file << "1" <<  "," <<"\t" << std::to_string(design_thrust_to_weight) <<  ",\t" << std::to_string(design_wing_loading) << std::endl;
        }
        else
        {
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(infile, line))
            {
                lines.push_back(line);
            }
            int last_design_point = std::stoi(lines[lines.size() - 1].substr(0, lines[lines.size() - 1].find("\t")));
            file << std::to_string(last_design_point + 1) << ",\t" << std::to_string(design_thrust_to_weight) << ",\t" << std::to_string(design_wing_loading) << std::endl;
        }
        file.close();
        infile.close();
    };
};

#endif
