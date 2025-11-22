/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef SRC_CONSTRAINT_ANALYSIS_PARSER_H_
#define SRC_CONSTRAINT_ANALYSIS_PARSER_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <atmosphere/atmosphere.h>
#include <aixml/node.h>

class readPolar
{
public:
    std::shared_ptr<node> polarXML = {};                       /** [-] The polar data in Polar Exchange File (PEF) format. */
    std::string flight_config = "Clean";                       /** [-] The flight configuration. */
    std::string interpolation = "linear";                      /** [-] The method of interpolation. */
    double machNumber = 0.2;                                   /** [-] The mach number of extracted polar. */
    std::vector<double> allowable_Mach = {};                   /** [-] The list of available mach numbers in the polar. */
    std::vector<double> CL_polar = {};                         /** [-] The lift coefficient vector. */
    std::vector<double> CD_polar = {};                         /** [-] The drag coefficient vector. */
    std::vector<double> alpha_polar = {};                      /** [-] The angle of attack vector. */
    double K1 = 0.0;                                           /** [-] The second order coefficient of quadratic drag fitting. */
    double K2 = 0.0;                                           /** [-] The first order coefficient of quadratic drag fitting. */
    double CD0 = 0.0;                                          /** [-] The zeroth order coefficient of quadratic drag fitting. */

    readPolar() = default;

    readPolar(
        const std::shared_ptr<node>& polarXML, 
        const std::string& flight_config, 
        const std::string& interpolation,
        const double& machNumber
        ) : polarXML(polarXML), flight_config(flight_config), interpolation(interpolation), machNumber(machNumber)
    {
        read_mach_numbers();
        if (allowable_Mach.size() < 2) 
        { 
            read_polar(allowable_Mach[0]); 
        }
        else 
        { 
            auto it = std::min_element(allowable_Mach.begin(), allowable_Mach.end(),
                [machNumber](double a, double b) {
                    return std::abs(a - machNumber) < std::abs(b - machNumber);
                });

            // Get the index of the closest element
            int index = std::distance(allowable_Mach.begin(), it);

            read_polar(allowable_Mach[index]);
        };
        std::vector<double> coeffs = quadratic_coefficients(CL_polar.back()*0.90);
        K1 = coeffs[0];
        K2 = coeffs[1];
        CD0 = coeffs[2];
    }

    auto getCD(const double& W_over_S, const double& n, const double& weight_fraction, const double& machNumber, const double& altitude) -> const double;

    auto calculateCD(const double& W_over_S, const double& n, const double& weight_fraction, const double& machNumber, const double& altitude, const double& AR, const double& e, const double& CD0) -> const double;

    auto getCL(const double& W_over_S, const double& n, const double& weight_fraction, const double& M, const double& h) -> const double;

    void read_polar(const double& machNumber);

    void read_mach_numbers();

    auto parser(const std::string& input_string) -> const std::vector<double>;

    auto interpCD(const double& CL) -> const double;

    auto interpalpha(const double& CL) -> const double;

    auto linear_interpolation(const double& x1, const double& x2, const double& y1, const double& y2, const double& x) -> const double;

    auto linear_interpolation(const std::vector<double> x, const std::vector<double> y, const double& xi) -> const double;

    auto quadratic_coefficients(const std::vector<double> x, const std::vector<double> y) -> const std::vector<double>;

    auto quadratic_coefficients() -> const std::vector<double>;

    auto quadratic_coefficients(const double factor) -> const std::vector<double>;
};

class readMission
{
public:
    const std::filesystem::path missionCSV;
    std::vector<double> altitude;
    std::vector<double> total_mass;
    std::vector<std::string> mode_name;

    readMission(const std::filesystem::path missionCSV) : missionCSV(missionCSV)
    {
        read_mission_data();
    }

    void read_mission_data();

    auto get_beta(const std::string segment, const double altitude) -> const double;

    auto get_beta(const std::string segment_from) -> const double;
};

#endif 