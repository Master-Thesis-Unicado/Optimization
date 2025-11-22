/*  Copyright (C) 2023 Chair of Aircraft Design, Technical University Munich
    This file is part of UNICADO.

    UNICADO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    UNICADO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with UNICADO.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @file aircraft_xml.cpp
 * @author Sebastian Oberschwendtner (sebastian.oberschwendtner@tum.de)
 * @brief The aircraft xml file interface.
 * @version 3.0.0
 * @date 2024-01-28
 */

/* === Includes === */
#include "aircraft_xml.h"
#include <string>
#include <memory>
#include <aixml/endnode.h>
#include <aircraftGeometry2/io/convert.h>

namespace io
{
    namespace detail
    {
        /**
         * @brief Create a endnode object and set its properties.
         *
         * @tparam T The value type of the endnode.
         * @param value The value of the endnode.
         * @param unit The unit of the endnode.
         * @param node_path The location in the parent node of the endnode.
         * @param description The description of the endnode.
         * @return Endnode<T> Return the created endnode.
         */
        template<typename T>
        auto create_endnode(const T value, const std::string& unit, const std::string& node_path, const std::string& description) -> Endnode<T>
        {
            /* Create the node an set its properties */
            Endnode<T> new_node{node_path, description};
            new_node.set_value(value);
            new_node.set_unit(unit);

            /* Return the new node */
            return new_node;
        }

    }; // namespace detail

    /**
     * @class AircraftXMLv3
     * @brief The xml interface for the aircraft xml file version 3.
     *
     */
    class AircraftXMLv3 : public detail::AircraftXMLInterface
    {
      public:
        /* === Constructors ===*/
        /**
         * @brief Construct a new AircraftXMLv3 object.
         *
         * @param aircraft_xml The aircraft xml data.
         */
        explicit AircraftXMLv3(std::shared_ptr<node> aircraft_xml) noexcept
            : aircraft_data(aircraft_xml)
        {
            /* Create the propulsion section if not existing */
            if (!this->aircraft_data->find("aircraft_exchange_file/sizing_point"))
            {
                /* Setup the first node level of the propulsion section */
                this->aircraft_data->operator[]("aircraft_exchange_file/sizing_point/thrust_to_weight") = "";
                this->aircraft_data->operator[]("aircraft_exchange_file/sizing_point/wing_loading") = "";

            }
        }

        /* === Destructor === */
        ~AircraftXMLv3() override = default;

        /* === Methods ===*/
        /**
         * @brief Insert the reference position into the xml.
         *
         * @param reference_position The reference position to insert into the xml.
         */
        void insert_1(const double thrust_to_weight) override
        {
            /* Set the node meta data */
            geom2::io::AixmlConverter::NodeInfo info;
            info.name = "thrust_to_weight";
            info.description = "Total thrust (kN) divided by maximum aircraft weight (kN)";

            /* Insert the point */
            this->insert_thrust_to_weight(
                this->aircraft_data->find("aircraft_exchange_file/sizing_point"),
                thrust_to_weight, info);
        }

        void insert_2(const double wing_loading) override
        {
            /* Set the node meta data */
            geom2::io::AixmlConverter::NodeInfo info;
            info.name = "wing_loading";
            info.description = "Maximum takeoff mass (MTOM) divided by wing area (Sref)";

            /* Insert the point */
            this->insert_wing_loading(
                this->aircraft_data->find("aircraft_exchange_file/sizing_point"),
                wing_loading, info);
        }

        /* === Methods ===*/
        /**
         * @brief Insert the reference position into the xml.
         *
         * @param reference_position The reference position to insert into the xml.
         */


    private:

        void insert_thrust_to_weight(node* target, const double thrust_to_weight, geom2::io::AixmlConverter::NodeInfo info)
        
        {
            node& thrust_to_weight_node = (*target)[info.name];
            thrust_to_weight_node.setAttrib("description", info.description);
            auto thrust_to_weight_value = detail::create_endnode(
                    thrust_to_weight, "1", "aircraft_exchange_file/sizing_point/thrust_to_weight",
                    "Total thrust (kN) divided by maximum aircraft weight (kN)");

            thrust_to_weight_value.update(thrust_to_weight_node);

        }

        void insert_wing_loading(node* target, const double wing_loading, geom2::io::AixmlConverter::NodeInfo info)
        {
            node& wing_loading_node = (*target)[info.name];
            wing_loading_node.setAttrib("description", info.description);
            auto wing_loading_value = detail::create_endnode(
                wing_loading, "kg/m^2", "aircraft_exchange_file/sizing_point/wing_loading",
                "Maximum takeoff mass (MTOM) divided by wing area (Sref)");

            wing_loading_value.update(wing_loading_node);

        }

        /* === Properties ===*/
        std::shared_ptr<node> aircraft_data; /** (-) The aircraft xml data. */
    };

    AircraftXML::AircraftXML(std::shared_ptr<node> aircraft_xml)
    {
        /* Select the fitting xml interface implementation */
        this->xml_interface = std::make_unique<AircraftXMLv3>(aircraft_xml);
    }

}; // namespace io
