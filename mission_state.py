# mission_state.py
from dataclasses import dataclass

@dataclass
class MissionState:
    time: float = 0.0        # [s]
    weight: float = 70000.0  # [kg]
    altitude: float = 0.0    # [m]
    speed: float = 0.0       # [m/s]
    distance: float = 0.0    # [m]
    fuel_used: float = 0.0   # [kg]
    segment_name: str = ""    
