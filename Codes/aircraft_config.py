# =========================================================================
# AIRCRAFT CONFIGURATION MODULE
# =========================================================================
"""
Centralized aircraft and mission configuration parameters.

This module contains all aircraft-wide configuration that is shared across
different mission phases (climb, cruise, descent). By centralizing these
parameters, we avoid duplication and ensure consistency across all modules.

Configuration includes:
- Aircraft physical parameters (mass, engines, reference area)
- Engine operational limits (Mach limits, altitude constraints)
- Atmospheric property calculations
- File paths for engine models
- Debug and system flags
"""

from __future__ import annotations
from atmosphere import Atmosphere

# ========= CONFIGURATION PARAMETERS ===========================================
class SystemConfiguration:
    """Centralized configuration management for the entire system."""
    
    # USER PATHS / SETTINGS
    ENGINE_STUB_PATH = r"D:/Icloud/iCloudDrive/Master Thesis/Mission Analysis Code/lls/stubs/engines/PW1127G-JM"

    # Aircraft / engines
    N_ENGINES        = 2
    S_REF_M2         = 118.36     # used for CLmax curve  
    
    # ========= WEIGHT BREAKDOWN COMPONENTS (SCALED AIRCRAFT - 0.5× ENGINE) =========
    # Operating Empty Weight (W_OE) components
    # Scaled from A320 baseline using power law scaling for 0.5× engine size
    W_AIRFRAME_KG    = 30000.0   # Weight of airframe structure (0.65× scaling)
    W_PROPULSION_KG  = 10000.0    # Weight of propulsion system (0.5× scaling - tied to engine)
    W_SYSTEMS_KG     = 10000.0    # Weight of systems (0.75× scaling - less size-dependent)
    
    # Payload weight estimation (scaled configuration)
    PAYLOAD_PER_PERSON_KG = 100.0  # 100 kg per person (passenger + luggage)
    DEFAULT_PASSENGERS = 108       # Scaled passenger capacity (0.6× of 180 = 108 seats)
    
    # Fuel weight (scaled for smaller aircraft)
    MAX_FUEL_KG      = 14500   # 14500 / Unicado Design Report Data 
    
    # Calculated weights
    @property
    def W_OE_KG(self):
        """Operating Empty Weight = Airframe + Propulsion + Systems"""
        return self.W_AIRFRAME_KG + self.W_PROPULSION_KG + self.W_SYSTEMS_KG
    
    @property
    def W_PL_KG(self):
        """Payload Weight = 100 kg × number of passengers"""
        return self.PAYLOAD_PER_PERSON_KG * self.DEFAULT_PASSENGERS
    
    @property
    def W_TO_KG(self):
        """Take-Off Weight = Operating Empty + Fuel + Payload"""
        return self.W_OE_KG + self.MAX_FUEL_KG + self.W_PL_KG
    
    # Engine / units / limits
    ENGINE_ALT_CLIP  = None      # meters; None = no clip
    M_MIN_DEFAULT    = 0.0        # From engine envelope: minimum operational Mach
    M_MIN_EFFECTIVE  = M_MIN_DEFAULT
    M_MMO            = 0.9392     # From engine envelope analysis: maximum operational Mach (at lever=1.0, altitude=10500 m)
    CL_MAX           = None       # Maximum lift coefficient 

    # DEBUG
    DEBUG = True  # Console printing

# ========= ATMOSPHERE INTEGRATION ================
class AtmosphericProperties:
    """Encapsulates atmospheric property calculations using the centralized atmosphere model."""
    
    # Physical constants
    G_C = Atmosphere.G_C  # Standard gravity [m/s²]
    
    def __init__(self):
        """Initialize the atmospheric properties calculator."""
        self._atmosphere = Atmosphere()
    
    def isa_properties(self, h_m: float):
        """Return (T [K], p [Pa], rho [kg/m^3]) for ISA using centralized atmosphere model."""
        return self._atmosphere.calculate_atmospheric_properties_meters(h_m)  # Direct meters input
    
    def a_from_altitude(self, h_m: float) -> float:
        """Speed of sound from altitude using centralized atmosphere model."""
        return self._atmosphere.get_speed_of_sound(h_m)

# Create global atmospheric properties instance for backward compatibility
_atmospheric_properties = AtmosphericProperties()

# ========= BACKWARD COMPATIBILITY EXPORTS ================
# Create global configuration instance
_config = SystemConfiguration()

# Expose configuration as module-level constants for easy access
ENGINE_STUB_PATH = _config.ENGINE_STUB_PATH
N_ENGINES = _config.N_ENGINES
S_REF_M2 = _config.S_REF_M2
ENGINE_ALT_CLIP = _config.ENGINE_ALT_CLIP
M_MIN_DEFAULT = _config.M_MIN_DEFAULT
M_MIN_EFFECTIVE = _config.M_MIN_EFFECTIVE
M_MMO = _config.M_MMO
CL_MAX = _config.CL_MAX
G_C = AtmosphericProperties.G_C
DEBUG = _config.DEBUG

# Weight breakdown components
W_AIRFRAME_KG = _config.W_AIRFRAME_KG
W_PROPULSION_KG = _config.W_PROPULSION_KG
W_SYSTEMS_KG = _config.W_SYSTEMS_KG
W_OE_KG = _config.W_OE_KG
W_PL_KG = _config.W_PL_KG
W_TO_KG = _config.W_TO_KG
INITIAL_MASS_KG = _config.W_TO_KG
MAX_FUEL_KG = _config.MAX_FUEL_KG
PAYLOAD_PER_PERSON_KG = _config.PAYLOAD_PER_PERSON_KG
DEFAULT_PASSENGERS = _config.DEFAULT_PASSENGERS

# Backward compatibility functions 
def isa_properties(h_m: float):
    """Return (T [K], p [Pa], rho [kg/m^3]) for ISA using centralized atmosphere model."""
    return _atmospheric_properties.isa_properties(h_m)

def a_from_altitude(h_m: float) -> float:
    """Speed of sound from altitude using centralized atmosphere model."""
    return _atmospheric_properties.a_from_altitude(h_m)