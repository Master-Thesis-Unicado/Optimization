/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#include <cmath>
#include <vector>
#include <numbers>
#include <math.h>
#include <numbers>
#include <atmosphere/atmosphere.h>
#include "../include/constraint_analysis/ca_functions.h"
#include "../include/constraint_analysis/ca_parser.h"

auto readPolar::calculateCD(const double& W_over_S, const double& n, const double& weight_fraction, const double& machNumber, const double& altitude, const double& AR, const double& e, const double& CD0) -> const double
{
    const double CL = readPolar::getCL(W_over_S, n, weight_fraction, machNumber, altitude);
    const double CD = CD0 + 1/(std::numbers::pi * AR * e) * pow(CL, 2);
    return CD;
}

auto readPolar::getCD(const double& W_over_S, const double& n, const double& weight_fraction, const double& machNumber, const double& altitude) -> const double
{
    const double CL = readPolar::getCL(W_over_S, n, weight_fraction, machNumber, altitude);
    return readPolar::interpCD(CL);
};

auto readPolar::getCL(const double& W_over_S, const double& n, const double& weight_fraction, const double& M, const double& h) -> const double
{
    atmosphere atm;
    double u = atm.getSpeedOfSound(h) * M;
    double rho = atm.getDensity(h);
    double CL = n * weight_fraction / (0.5 * rho * pow(u, 2.0)) * W_over_S;
    return CL;
};

void readPolar::read_polar(const double& machNumber)
{
    auto configuration(this->polarXML->at("configurations/configuration@" + flight_config + "/polars"));
    for (int num_polar(0); num_polar < configuration.getVector("polar").size(); num_polar++) {
        auto polar(&configuration.at("polar@" + std::to_string(num_polar + 1)));
        const double Mach((*polar).at("machnumber"));
        if (Mach == machNumber)
        {
            this->alpha_polar = readPolar::parser(std::string((*polar).at("Alpha")));
            this->CL_polar = readPolar::parser(std::string((*polar).at("Cl")));
            this->CD_polar = readPolar::parser(std::string((*polar).at("Cd")));
        }
    }
};

void readPolar::read_mach_numbers()
{
    auto configuration(this->polarXML->at("configurations/configuration@" + flight_config + "/polars"));
    for (int num_polar(0); num_polar < configuration.getVector("polar").size(); num_polar++) {
        auto polar(&configuration.at("polar@" + std::to_string(num_polar + 1)));
        const double Mach((*polar).at("machnumber"));
        this->allowable_Mach.push_back(Mach);
    }
};


auto readPolar::parser(const std::string& input_string) -> const std::vector<double>
{
    std::vector<double> output;
    std::stringstream ss(input_string);
    std::string temp;
    while (std::getline(ss, temp, ';')) {
        try {
            output.push_back(std::stod(temp));
        }
        catch (...)
        {
            throw ("parser error occured");
        };
    }
    return output;
}

auto readPolar::interpCD(const double& CL) -> const double 
{
    double CD = 0.0;
    if(CL<this->CL_polar.back())
    { 
        CD = readPolar::linear_interpolation(this->CL_polar, this->CD_polar, CL); 
    }
    else {
        CD = this->K1 * pow(CL, 2) + this->K2 * CL + this->CD0;
    };
    return CD;
}

auto readPolar::interpalpha(const double& CL) -> const double
{
    const double alpha = readPolar::linear_interpolation(this->CL_polar, this->alpha_polar, CL);
    return alpha;
}

auto readPolar::linear_interpolation(const double& x1, const double& x2, const double& y1, const double& y2, const double& x) -> const double
{
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

auto readPolar::linear_interpolation(const std::vector<double> x, const std::vector<double> y, const double& xi) -> const double
{
    auto lb = std::lower_bound(x.begin(), x.end(), xi);
    auto idx = std::distance(x.begin(), lb);
    return y[idx-1] + (xi - x[idx-1]) * (y[idx] - y[idx-1]) / (x[idx] - x[idx-1]);
}

auto readPolar::quadratic_coefficients(const std::vector<double> x, const std::vector<double> y) -> const std::vector<double>
{
    int n = x.size();
    double A_x4 = 0, A_x3 = 0, A_x2 = 0, A_x1 = 0, A_x0 = n, b_x2y = 0, b_xy = 0, b_y = 0;
    for (int i = 0; i < n; ++i)
    {
        A_x4 += pow(x[i], 4);
        A_x3 += pow(x[i], 3);
        A_x2 += pow(x[i], 2);
        A_x1 += x[i];
        b_x2y += pow(x[i], 2) * y[i];
        b_xy += x[i] * y[i];
        b_y += y[i];
    }

    double det =
        A_x4 * (A_x2 * A_x0 - A_x1 * A_x1)
        - A_x3 * (A_x3 * A_x0 - A_x1 * A_x2)
        + A_x2 * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_a = 
        b_x2y * (A_x2 * A_x0 - A_x1 * A_x1)
        - b_xy * (A_x3 * A_x0 - A_x1 * A_x2)
        + b_y * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_b = 
        A_x4 * (b_xy * A_x0 - b_y * A_x1)
        - A_x3 * (b_x2y * A_x0 - b_y * A_x2)
        + A_x2 * (b_x2y * A_x1 - b_xy * A_x2);

    double det_c = 
        A_x4 * (A_x2 * b_y - b_xy * A_x1)
        - A_x3 * (A_x3 * b_y - b_x2y * A_x1)
        + A_x2 * (A_x3 * b_xy - b_x2y * A_x2);

    return std::vector<double>{det_a / det, det_b / det, det_c / det};
}

auto readPolar::quadratic_coefficients(const double factor) -> const std::vector<double>
{
    std::vector<double> x_init = this->CL_polar;
    std::vector<double> y_init = this->CD_polar;
    auto it = std::distance(x_init.begin(), std::lower_bound(x_init.begin(), x_init.end(), factor));
    std::vector<double> x(x_init.begin() + it, x_init.end());
    std::vector<double> y(y_init.begin() + it, y_init.end());

    int n = x.size();
    double A_x4 = 0, A_x3 = 0, A_x2 = 0, A_x1 = 0, A_x0 = n, b_x2y = 0, b_xy = 0, b_y = 0;
    for (int i = 0; i < n; ++i)
    {
        A_x4 += pow(x[i], 4);
        A_x3 += pow(x[i], 3);
        A_x2 += pow(x[i], 2);
        A_x1 += x[i];
        b_x2y += pow(x[i], 2) * y[i];
        b_xy += x[i] * y[i];
        b_y += y[i];
    }

    double det =
        A_x4 * (A_x2 * A_x0 - A_x1 * A_x1)
        - A_x3 * (A_x3 * A_x0 - A_x1 * A_x2)
        + A_x2 * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_a =
        b_x2y * (A_x2 * A_x0 - A_x1 * A_x1)
        - b_xy * (A_x3 * A_x0 - A_x1 * A_x2)
        + b_y * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_b =
        A_x4 * (b_xy * A_x0 - b_y * A_x1)
        - A_x3 * (b_x2y * A_x0 - b_y * A_x2)
        + A_x2 * (b_x2y * A_x1 - b_xy * A_x2);

    double det_c =
        A_x4 * (A_x2 * b_y - b_xy * A_x1)
        - A_x3 * (A_x3 * b_y - b_x2y * A_x1)
        + A_x2 * (A_x3 * b_xy - b_x2y * A_x2);

    return std::vector<double>{det_a / det, det_b / det, det_c / det};
}

auto readPolar::quadratic_coefficients() -> const std::vector<double>
{
    std::vector<double> x = this->CL_polar;
    std::vector<double> y = this->CD_polar;

    int n = x.size();
    double A_x4 = 0, A_x3 = 0, A_x2 = 0, A_x1 = 0, A_x0 = n, b_x2y = 0, b_xy = 0, b_y = 0;
    for (int i = 0; i < n; ++i)
    {
        A_x4 += pow(x[i], 4);
        A_x3 += pow(x[i], 3);
        A_x2 += pow(x[i], 2);
        A_x1 += x[i];
        b_x2y += pow(x[i], 2) * y[i];
        b_xy += x[i] * y[i];
        b_y += y[i];
    }

    double det =
        A_x4 * (A_x2 * A_x0 - A_x1 * A_x1)
        - A_x3 * (A_x3 * A_x0 - A_x1 * A_x2)
        + A_x2 * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_a =
        b_x2y * (A_x2 * A_x0 - A_x1 * A_x1)
        - b_xy * (A_x3 * A_x0 - A_x1 * A_x2)
        + b_y * (A_x3 * A_x1 - A_x2 * A_x2);

    double det_b =
        A_x4 * (b_xy * A_x0 - b_y * A_x1)
        - A_x3 * (b_x2y * A_x0 - b_y * A_x2)
        + A_x2 * (b_x2y * A_x1 - b_xy * A_x2);

    double det_c =
        A_x4 * (A_x2 * b_y - b_xy * A_x1)
        - A_x3 * (A_x3 * b_y - b_x2y * A_x1)
        + A_x2 * (A_x3 * b_xy - b_x2y * A_x2);

    return std::vector<double>{det_a / det, det_b / det, det_c / det};
}

void readMission::read_mission_data()
{
    std::ifstream file(this->missionCSV);

    std::string line;

    std::getline(file, line);
    std::istringstream header_stream(line);
    std::string column;
    std::vector<std::string> headers;
    int altitudeIndex = -1, massIndex = -1, modeIndex = -1, index = 0;

    while (std::getline(header_stream, column, ';')) {
        headers.push_back(column);
        if (column == " Altitude [m]") altitudeIndex = index;
        if (column == " Total mass [kg]") massIndex = index;
        if (column == " Mode name [-]") modeIndex = index;
        index++;
    }

    // Read the data
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string value;
        std::vector<std::string> row;
        int colIndex = 0;

        while (std::getline(line_stream, value, ';')) {
            row.push_back(value);
        }

        if (row.size() > std::max({ altitudeIndex, massIndex, modeIndex })) {
            try {
                this->altitude.push_back(std::stod(row[altitudeIndex]));
                this->total_mass.push_back(std::stod(row[massIndex]));
                this->mode_name.push_back(row[modeIndex]);
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Skipping invalid row due to conversion error: " << line << std::endl;
            }
        }
    }
};

auto readMission::get_beta(const std::string segment, const double altitude) -> const double
{
    for (size_t i = 0; i < this->altitude.size(); ++i) {
        if (this->altitude[i] >= altitude && this->mode_name[i] == segment) {
            return this->total_mass[i]/this->total_mass[0];
        }
    }
    return 0.98;
};

auto readMission::get_beta(const std::string segment_from) -> const double
{
    for (size_t i = 1; i < this->mode_name.size(); ++i) {
        if (this->mode_name[i - 1] == segment_from && this->mode_name[i] != segment_from) {
            return this->total_mass[i] / this->total_mass[0];
        }
    }
    return 0.98;
};