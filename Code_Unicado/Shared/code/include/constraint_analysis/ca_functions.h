/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

#ifndef CONSTRAINTANALYSIS_CONSTRAINTANALYSIS_H_
#define CONSTRAINTANALYSIS_CONSTRAINTANALYSIS_H_

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <atmosphere/atmosphere.h>
#include <memory>
#include <moduleBasics/module.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace Mattingly {

    class constraint_analysis {
    public:
        atmosphere atm;
        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D_data Vector of the drag coefficients for the corresponding wing loadings.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         */
        auto constant_altitude_speed_cruise(const std::vector<double>& W_over_S_data, const std::vector<double>& C_D_data, double weight_fraction, double thrust_lapse, double Mach, double altitude) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D_data Vector of the drag coefficients for the corresponding wing loadings.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param const_speed_climb_rate Value of the climb rate.
         */
        auto constant_speed_climb(const std::vector<double>& W_over_S_data, const std::vector<double>& C_D_data, double weight_fraction, double thrust_lapse, double Mach, double altitude, double const_speed_climb_rate) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D0 Value of the zero-lift drag.
         * @param K_1 Value of the K1-factor.
         * @param K_2 Value of the K2-factor.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param n Value of the load factor.
         */
        auto constant_altitude_speed_turn(const std::vector<double>& W_over_S_data, double K_1, double K_2, double C_D0, double weight_fraction, double thrust_lapse, double Mach, double altitude, double n) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D_data Vector of the drag coefficients for the corresponding wing loadings.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param horizontal_acceleration Value of the horizontal_acceleration.
         */
        auto horizontal_acceleration(const std::vector<double>& W_over_S_data, const std::vector<double>& C_D_data, double weight_fraction, double thrust_lapse, double Mach, double altitude, double horizontal_acceleration) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D_data Vector of the drag coefficients for the corresponding wing loadings.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param s_G Value of the takeoff distance.
         * @param CL_max_TO Value of the lift coefficient at takeoff.
         */
        auto takeoff_ground_roll(const std::vector<double>& W_over_S_data, double weight_fraction, double thrust_lapse, double Mach, double altitude, double s_G, double CL_max_TO) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D_data Vector of the drag coefficients for the corresponding wing loadings.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param my_B Value of the braking coefficient.
         * @param s_G Value of the takeoff distance.
         */
        auto braking_roll(double C_D_L, double C_L_L, double weight_fraction, double thrust_lapse, double Mach, double altitude, double my_B, double s_G) -> const double;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D0 Value of the zero-lift drag.
         * @param K_1 Value of the K1-factor.
         * @param K_2 Value of the K2-factor.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param alt_sc Value of service ceiling altitude.
         */
        auto service_ceiling(const std::vector<double>& W_over_S_data, double K_1, double K_2, double C_D0, double weight_fraction, double thrust_lapse, double Mach, double altitude, double alt_sc) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param W_over_S_data Vector of the wing loading.
         * @param C_D0 Value of the zero-lift drag.
         * @param K_1 Value of the K1-factor.
         * @param K_2 Value of the K2-factor.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         * @param thrust_lapse Value of the thrust lapse of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param CL_max_TO Value of the lift coefficient at takeoff.
         * @param takeoff_climb_angle Value of the take-off climb angle.
         */
        auto takeoff_climb_angle(const std::vector<double>& W_over_S_data, const std::vector<double>& C_D_data, double weight_fraction, double thrust_lapse, double Mach, double altitude, double CL_max_TO, double takeoff_climb_angle) -> const std::vector<double>;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param CL_alpha Value of the lift slope.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param V_TO_L Value of speed at takeoff and landing.
         * @param w_g Value of the vertical gust speed.
         * @param dn_G Value of the delta gust load factor.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         */
        auto gust(double CL_alpha, double altitude, double V_TO_L, double w_g, double dn_G, double weight_fraction) -> const double;

        /**
         * @brief Calculates and outputs thrust-to-weight ratios (T/W) 
         *        for a range of wing loadings (W/S) and drag coefficients (C_D)
         *        during constant altitude and speed cruise.
         * @param CL_max Value of the maximum lift coefficient.
         * @param altitude Value of the altitude of the corresponding flight segment.
         * @param Mach Value of the mach number of the corresponding flight segment.
         * @param weight_fraction Value of the weight fraction of the corresponding flight segment.
         */
        auto min_speed(double CL_max, double altitude, double Mach,  double weight_fraction) -> const double;

    private:

        // Membervariables
        double rate_of_climb_SC = 0.5; //rest climb capability at service ceiling in m/s
        double g_0 = 9.81;
        double rho = atm.getDensity(0);
        double k_T0 = 1.1; //velocity ratio at takeoff (V_TO = k_TO*V_STALL) CS25

        std::shared_ptr<node> configuration_xml;   
        std::shared_ptr<node> aircraft_xml;
    };

} // namespace Mattingly

#endif // CONSTRAINTANALYSIS_CONSTRAINTANALYSIS_H_
