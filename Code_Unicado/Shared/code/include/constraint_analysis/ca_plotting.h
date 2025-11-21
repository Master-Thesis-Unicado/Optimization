/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/


#ifndef SRC_CONSTRAINT_ANALYSIS_PLOTTING_H_
#define SRC_CONSTRAINT_ANALYSIS_PLOTTING_H_

#include <cmath>
#include <vector>
#include <numbers>
#include <sstream>
#include <string>
#include <format>
#include <iostream>
#include <atmosphere/atmosphere.h>
#include <aixml/node.h>
#include <matplot/matplot.h>

class ConstraintAnalysis_Plot
{
public:
    std::shared_ptr<matplot::figure_type> figure = matplot::figure(true);
    std::vector<std::string> legend{};

    ConstraintAnalysis_Plot()
    {
        auto axis = figure->current_axes();
        figure->size(800, 600);
        figure->position({ 0, 0, 800, 600 });
        axis->hold(true);
        axis->grid(true);
        axis->font_size(18);
        axis->xlabel("Wing Loading [N/m^2]");
        axis->x_axis().label_font_size(18);
        axis->ylabel("Thrust to Weight [-]");
        axis->y_axis().label_font_size(18);
        axis->ylim({ 0,1 });
    };

    void add_curve(std::vector<double> x, std::vector<double> y, std::string label)
    {
        auto axis = figure->current_axes();
        if (y.size() > 5) {
            axis->plot(x, y)->line_width(2);
        } else { 
        // Format the number with two decimal places (or any desired precision)
        axis->scatter(x, y)->marker(".").marker_size(5);
        std::string str = "W/S = " + std::format("{:.2f}", x[0] / 9.81) + "\\nT/W = " + std::format("{:.4f}", y[0]);
        y[0] += 0.05;
        axis->text(x, y, str);
        }
        legend.push_back(label);
    };

    void add_curve(double x_lim, std::string label)
    {
        std::vector<double> x = { x_lim, x_lim };
        std::vector<double> y = { 0, 10 };
        auto axis = figure->current_axes();
        axis->plot(x, y)->line_width(2);
        legend.push_back(label);
    };

    void save_figure(std::filesystem::path plot_dir)
    {
        try {
            /* Save the finished plot */
            auto axis = figure->current_axes();
            axis->xlim({ 600.0, 8000.0 });
            axis->legend(legend);
            axis->legend()->location(matplot::legend::general_alignment::topleft);
            auto plot_path_string = plot_dir.string();
            std::replace(plot_path_string.begin(), plot_path_string.end(), '\\', '/');
            figure->save(plot_path_string);
        } catch (const std::exception& e) {
            std::cerr << "Error: Could not save plot: " << e.what() << std::endl;
        }
    };

    void fill_infeasible_area(std::vector<double> x, std::vector<double> y , std::vector<double> xlims)
    {
        int i = 0;
        std::sort(xlims.begin(), xlims.end());
        for (double y_elem : y) {
            if (x[i] <= xlims[0] || x[i] >= xlims[1]) { y[i] = 100.0; };
            i++;
        }
        auto axis = figure->current_axes();
        x.push_back(x.back());
        x.push_back(x.front());
        x.push_back(x.front());
        y.push_back(0.0);
        y.push_back(0.0);
        y.push_back(y.front());
        const std::array<float, 4> rgba_color = { 0.7f, 1.0f, 0.0f, 0.0f };
        axis->area(x, y)->face_color(rgba_color);
        legend.push_back("Infeasible Area");
    }
};

#endif