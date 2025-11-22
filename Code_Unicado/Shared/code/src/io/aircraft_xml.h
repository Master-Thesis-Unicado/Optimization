#ifndef SRC_IO_AIRCRAFT_XML_H_
#define SRC_IO_AIRCRAFT_XML_H_

/* === Includes === */
#include "../mission_analysis.h"
#include "aixml/node.h"
#include <functional>
#include <memory>

namespace io
{
    namespace detail
    {
        /**
         * @brief get xml data
         */
        struct AircraftXMLInterface
        {
            virtual ~AircraftXMLInterface() = default;

            // Method for inserting data
            virtual void insert_1(double thrust_to_weight) = 0;
            virtual void insert_2(double wing_loading) = 0;
        };
    }; // namespace detail

    /**
     * @class AircraftXML
     * @brief Die main interface to aircraft.
     *
     */
    class AircraftXML
    {
    public:
        /**
         * @brief Constructor for ACxml.
         * @param aircraft_xml
         */
        explicit AircraftXML(std::shared_ptr<node> aircraft_xml);

        void insert_1(const double &thrust_to_weight)
        {
            this->xml_interface->insert_1(thrust_to_weight);
        }

        void insert_2(const double &wing_loading)
        {
            this->xml_interface->insert_2(wing_loading);
        }

    private:
        std::unique_ptr<detail::AircraftXMLInterface> xml_interface;  // Pointer for interface
    };
}; // namespace io

#endif // SRC_IO_AIRCRAFT_XML_H_
