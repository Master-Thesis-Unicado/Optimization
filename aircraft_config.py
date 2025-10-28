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
- File paths for aerodynamic data and engine models
- Debug and system flags
"""

from __future__ import annotations
from atmosphere import Atmosphere

# ========= CONFIGURATION PARAMETERS ===========================================
class SystemConfiguration:
    """Centralized configuration management for the entire system."""
    
    def __init__(self):
        """Initialize the configuration and set calculated values."""
        # Set INITIAL_MASS_KG to the calculated take-off weight
        self.INITIAL_MASS_KG = self.W_TO_KG
    
    @classmethod
    def get_initial_mass_kg(cls):
        """Get the initial mass as a class method for backward compatibility."""
        instance = cls()
        return instance.W_TO_KG
    
    # USER PATHS / SETTINGS
    AERO_XLSX  = r"D:\Icloud\iCloudDrive\Master Thesis\Mission Analysis Code\Aero\Ps Curves.xlsx"
    AERO_SHEET = "Sheet4"
    ENGINE_STUB_PATH = r"D:/Icloud/iCloudDrive/Master Thesis/Mission Analysis Code/lls/stubs/engines/PW1127G-JM"

    # Aircraft / engines
    N_ENGINES        = 2
    S_REF_M2         = 122.4     # used for CLmax curve (can be overridden from Excel)
    
    # ========= WEIGHT BREAKDOWN COMPONENTS (A320 SPECIFICATIONS) =========
    # Operating Empty Weight (W_OE) components
    W_AIRFRAME_KG    = 15620.0   # Weight of airframe structure (A320)
    W_PROPULSION_KG  = 11000.0   # Weight of propulsion system (A320 engines + nacelles)
    W_SYSTEMS_KG     = 7000.0    # Weight of systems (A320 avionics, hydraulics, etc.)
    
    # Payload weight estimation (A320 typical configuration)
    PAYLOAD_PER_PERSON_KG = 100.0  # 100 kg per person (passenger + luggage)
    DEFAULT_PASSENGERS = 180       # A320 typical passenger capacity (180-200 seats)
    
    # Fuel weight (A320 specifications)
    MAX_FUEL_KG      = 5000   # 
    
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
    
    def current_mass_kg(self, fuel_remaining_kg: float = None) -> float:
        """
        Calculate current aircraft mass based on fuel remaining.
        
        Args:
            fuel_remaining_kg: Current fuel remaining in kg. If None, uses max fuel.
            
        Returns:
            Current total aircraft mass in kg
        """
        if fuel_remaining_kg is None:
            fuel_remaining_kg = self.MAX_FUEL_KG
        
        return self.W_OE_KG + fuel_remaining_kg + self.W_PL_KG
    
    def fuel_burned_kg(self, fuel_remaining_kg: float) -> float:
        """Calculate how much fuel has been burned."""
        return self.MAX_FUEL_KG - fuel_remaining_kg
    
    # Backward compatibility - use calculated take-off weight
    INITIAL_MASS_KG  = None  # Will be set to W_TO_KG in __init__

    # Engine / units / limits
    ENGINE_ALT_CLIP  = None      # meters; None = no clip
    M_MIN_DEFAULT    = 0.0        # From engine envelope: minimum operational Mach
    M_MIN_EFFECTIVE  = M_MIN_DEFAULT
    M_MMO            = 0.94       # From engine envelope: maximum operational Mach (may be overridden from Excel)
    CL_MAX           = None       # Will be set from Excel cell 

    # DEBUG / PARAM APPLICATION
    DEBUG = True # Console printing
    AUTO_APPLY_PARAMS_FROM_EXCEL = True  # when True, scalars found in Excel override module-level defaults

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
AERO_XLSX = _config.AERO_XLSX
AERO_SHEET = _config.AERO_SHEET
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
AUTO_APPLY_PARAMS_FROM_EXCEL = _config.AUTO_APPLY_PARAMS_FROM_EXCEL

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
