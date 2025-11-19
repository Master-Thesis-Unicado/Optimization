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
#include <engine/engine.h>
#include "simple_mission_analysis.h"

class Canvas {
public:
	std::shared_ptr<matplot::figure_type> figure = matplot::figure(true);
	std::vector<std::string> legend{};
    std::shared_ptr<matplot::axes_type> axis = figure->current_axes();
    std::string path = "";

    Canvas(std::string path) : path(path) {
        initialize_plot();
        figure->position({ 0, 0, 800, 600 });
    };

	void initialize_plot() {
        axis->hold(true);
        axis->grid(true);
        axis->font_size(18);
        axis->x_axis().label_font_size(18);
        axis->y_axis().label_font_size(18);
	}

    void add_data(std::vector<double> x_axis, std::vector<double> y_axis) {
        axis->plot(x_axis, y_axis)->line_width(2);
    }

    void save_canvas() {
        try {
            figure->save(path);
        }
        catch (const std::exception& e) {
            std::cerr << "Error: Could not save plot: " << e.what() << std::endl;
        }
    }

    void set_axis_titles(std::string x_label, std::string y_label) {
        axis->xlabel(x_label);
        axis->ylabel(y_label);
    }
};
