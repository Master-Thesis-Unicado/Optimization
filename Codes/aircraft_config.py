# ========================================================================
# AIRCRAFT CONFIGURATION MODULE
# ========================================================================
"""
Aircraft physical parameters and operational limits.

Centralized configuration for mission analysis:
    - Geometric parameters: S_ref, N_engines
    - Mass breakdown: m_OE, m_PL, m_fuel, m_TO
    - Operational limits: M_min, M_MMO, CL_max, h_max
    - Fuel system: tank CG positions, density
    - Atmospheric models: ISA properties

Provides backward-compatible exports for all modules.
"""

from __future__ import annotations


# ========================================================================
# SYSTEM CONFIGURATION
# ========================================================================

class SystemConfiguration:
    """Aircraft physical parameters and operational configuration."""
    
    # ────────────────────────────────────────────────────────────────────
    # Geometric Parameters
    # ────────────────────────────────────────────────────────────────────
    
    N_ENGINES = 2             # N_eng [-]: number of engines
    S_REF_M2 = 118.36         # S_ref [m²]: wing reference area
    
    
    # ────────────────────────────────────────────────────────────────────
    # Mass Breakdown
    # ────────────────────────────────────────────────────────────────────
    
    # Operating Empty Weight (m_OE) components
    W_AIRFRAME_KG = 35850.0   # m_airframe [kg]: airframe structure
    W_PROPULSION_KG = 11950.0 # m_propulsion [kg]: propulsion system
    W_SYSTEMS_KG = 9554.0     # m_systems [kg]: avionics and systems
    
    # Payload parameters
    PAYLOAD_PER_PERSON_KG = 80  # m_pax [kg]: mass per passenger
    DEFAULT_PASSENGERS = 60     # N_pax [-]: passenger capacity
    
    # Fuel capacity
    W_FUEL_KG = 14500   # m_fuel,max [kg]: maximum fuel capacity
    
    # Computed mass properties
    @property
    def W_OE_KG(self):
        """m_OE [kg]: Operating Empty Weight = m_airframe + m_propulsion + m_systems."""
        return self.W_AIRFRAME_KG + self.W_PROPULSION_KG + self.W_SYSTEMS_KG
    
    @property
    def W_PL_KG(self):
        """m_PL [kg]: Payload mass = m_pax · N_pax."""
        return self.PAYLOAD_PER_PERSON_KG * self.DEFAULT_PASSENGERS
    
    @property
    def W_TO_KG(self):
        """m_TO [kg]: Takeoff mass = m_OE + m_fuel + m_PL."""
        return self.W_OE_KG + self.W_FUEL_KG + self.W_PL_KG
    
    
    # ────────────────────────────────────────────────────────────────────
    # Operational Limits
    # ────────────────────────────────────────────────────────────────────
    
    # Altitude limits
    ENGINE_ALT_CLIP = None      # h_clip [m]: altitude clipping (None = no limit)
    
    # Mach number limits
    M_MIN_DEFAULT = 0.0         # M_min,default [-]: nominal minimum Mach
    M_MIN_EFFECTIVE = 0.2       # M_min,eff [-]: effective minimum operational Mach
    M_MMO = 0.9392              # M_MMO [-]: maximum operating Mach number
    
    # Aerodynamic limits
    CL_MAX = 1.6                # CL_max [-]: maximum lift coefficient
    
    # Throttle lever limits (physical constraint)
    LEVER_MIN = 0.0             # δ_min [-]: minimum throttle position (idle)
    LEVER_MAX = 1.0             # δ_max [-]: maximum throttle position (full)
    
    
    # ────────────────────────────────────────────────────────────────────
    # Fuel System Configuration
    # ────────────────────────────────────────────────────────────────────
    
    # Fuel properties
    KEROSENE_DENSITY_KGPM3 = 0.8  # ρ_fuel [kg/m³]: kerosene density
    
    # Tank center of gravity positions: x_CG [m] in body frame
    TANK_CG_POSITIONS = {
        0: 14.88,  # Tank 0: Inner Left
        1: 16.09,  # Tank 1: Outer Left
        2: 14.88,  # Tank 2: Inner Right
        3: 16.09,  # Tank 3: Outer Right
        4: 14.99   # Tank 4: Center Wing
    }
    
    # Tank identifiers
    TANK_NAMES = {
        0: "Inner Left",
        1: "Outer Left",
        2: "Inner Right",
        3: "Outer Right",
        4: "Center Wing"
    }
    
    # Aircraft center of gravity
    ZERO_FUEL_CG_X = 16.208                  # x_CG,ZFW [m]: CG position at zero fuel weight (OEW + Payload, no fuel)
    DEFAULT_COG_LOCATION = [ZERO_FUEL_CG_X, 0.0, 0.0]  # [x, y, z] [m]: default CG position (x-component references ZERO_FUEL_CG_X)
    
    # Diagnostic configuration
    FUEL_LEVEL_PRINT_ENABLED = True
    FUEL_LEVEL_PRINT_SAMPLE_RATE = 10  # Sample rate for fuel level output


# ========================================================================
# MODULE EXPORTS
# ========================================================================

# Configuration instance
_config = SystemConfiguration()

# Atmospheric models: ISA properties T(h), ρ(h), p(h), a(h), g_c
from atmosphere import isa_properties, a_from_altitude, G_C, Atmosphere

# Atmospheric properties interface
_atmospheric_properties = Atmosphere()

# ────────────────────────────────────────────────────────────────────
# Module-level exports for backward compatibility
# ────────────────────────────────────────────────────────────────────

# Geometric parameters
N_ENGINES = _config.N_ENGINES           # N_eng [-]
S_REF_M2 = _config.S_REF_M2             # S_ref [m²]

# Mass components
W_AIRFRAME_KG = _config.W_AIRFRAME_KG   # m_airframe [kg]
W_PROPULSION_KG = _config.W_PROPULSION_KG  # m_propulsion [kg]
W_SYSTEMS_KG = _config.W_SYSTEMS_KG     # m_systems [kg]
W_OE_KG = _config.W_OE_KG               # m_OE [kg]
W_PL_KG = _config.W_PL_KG               # m_PL [kg]
W_TO_KG = _config.W_TO_KG               # m_TO [kg]
INITIAL_MASS_KG = _config.W_TO_KG       # m_0 [kg] (alias)
W_FUEL_KG = _config.W_FUEL_KG           # m_fuel,max [kg]
PAYLOAD_PER_PERSON_KG = _config.PAYLOAD_PER_PERSON_KG  # m_pax [kg]
DEFAULT_PASSENGERS = _config.DEFAULT_PASSENGERS        # N_pax [-]

# Operational limits
ENGINE_ALT_CLIP = _config.ENGINE_ALT_CLIP     # h_clip [m]
M_MIN_DEFAULT = _config.M_MIN_DEFAULT         # M_min,default [-]
M_MIN_EFFECTIVE = _config.M_MIN_EFFECTIVE     # M_min,eff [-]
M_MMO = _config.M_MMO                         # M_MMO [-]
CL_MAX = _config.CL_MAX                       # CL_max [-]
LEVER_MIN = _config.LEVER_MIN                 # δ_min [-]
LEVER_MAX = _config.LEVER_MAX                 # δ_max [-]

# Fuel system
KEROSENE_DENSITY_KGPM3 = _config.KEROSENE_DENSITY_KGPM3  # ρ_fuel [kg/m³]
TANK_CG_POSITIONS = _config.TANK_CG_POSITIONS             # x_CG,tank [m]
TANK_NAMES = _config.TANK_NAMES                           # Tank identifiers
DEFAULT_COG_LOCATION = _config.DEFAULT_COG_LOCATION       # [x,y,z]_CG [m]
ZERO_FUEL_CG_X = _config.ZERO_FUEL_CG_X                   # x_CG,ZFW [m]

# Diagnostic flags
FUEL_LEVEL_PRINT_ENABLED = _config.FUEL_LEVEL_PRINT_ENABLED
FUEL_LEVEL_PRINT_SAMPLE_RATE = _config.FUEL_LEVEL_PRINT_SAMPLE_RATE

# Debug flag for conditional logging (disabled by default)
DEBUG = False