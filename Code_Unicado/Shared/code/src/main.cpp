/*  Copyright (C) MTan, MHux, TWec
    This file is part of SHADOWCADO.

    SHADOWCADO is NOT a free software: 1000 EUR
*/

/* === Includes === */
#include "mission_analysis.h"
#include "toolinfo.h"

/* === Main === */
/**
 * @brief Main executable of the propulsionDesign tool.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 * @return int The exit code of the tool.
 */
int main(int argc, char **argv)
{
    try{
        DefaultAnalysis testanalysis(argc, argv, TOOL_NAME, TOOL_VERSION);
        return testanalysis.execute();
    } catch (const std::string &error) {
        std::cerr << error << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
    }
}

