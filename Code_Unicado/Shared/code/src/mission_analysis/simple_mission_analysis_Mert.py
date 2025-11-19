import math
from dataclasses import dataclass
from enum import Enum


# Placeholder for the atmosphere model
import math

class Atmosphere:
    def get_temperature(self, altitude):
        # ISA model formula: T(h) = T0 + lapse_rate * h
        # See theory here: ./atmosphere_theory.md
        T0 = 288.15  # Sea level temperature in Kelvin
        lapse_rate = -0.0065  # K/m
        return T0 + lapse_rate * altitude

    def get_speed_of_sound(self, altitude):
        # Speed of sound: a = sqrt(gamma * R * T)
        # See theory here: ./atmosphere_theory.md
        gamma = 1.4
        R = 287.05  # J/(kg·K)
        temperature = self.get_temperature(altitude)
        return math.sqrt(gamma * R * temperature)


atm = Atmosphere()

# Constants
k_TO = 1.1
g0 = 9.81
a_standard = atm.get_speed_of_sound(0.0)
friction_coeff = 0.05


class SegmentType(Enum):
    Accelerated_Climb = 1
    Subsonic_Loiter = 2
    Idle = 3
    Constant_Altitude_Speed_Cruise = 4


@dataclass
class Segment:
    type: SegmentType
    TSFC: float
    Mach: float
    CL: float
    CD: float
    delta_t: float
    altitude: float
    dh_dt: float = 0.0
    dv_dt: float = 0.0
    initial_weight: float = 1.0
    thrust: float = 1.0
    weight_fraction: float = 1.0


def accelerated_climb(segment: Segment) -> Segment:
    T = atm.get_temperature(segment.altitude)
    T0 = atm.get_temperature(0.0)
    nondim_temperature = T / T0
    a = atm.get_speed_of_sound(segment.altitude)
    delta_energy = segment.dh_dt * segment.delta_t + (
        (segment.Mach * a + segment.dv_dt * segment.delta_t) ** 2 - (segment.Mach * a) ** 2
    ) / (2 * g0)

    exponent = -segment.TSFC * g0 / (segment.Mach * math.sqrt(nondim_temperature) * a_standard) * delta_energy
    exponent /= (1 - (segment.CD / segment.CL) * (segment.initial_weight / segment.thrust))
    segment.weight_fraction = math.exp(exponent)
    return segment


def subsonic_loiter(segment: Segment) -> Segment:
    EF = segment.CL / (segment.CD * segment.TSFC * g0)
    segment.weight_fraction = math.exp(-segment.delta_t / EF)
    return segment


def idle_thrust(segment: Segment, W: float) -> Segment:
    segment.weight_fraction = 1.0 - segment.TSFC * g0 * segment.thrust * segment.delta_t / W
    return segment


def constant_altitude_cruise(segment: Segment) -> Segment:
    T = atm.get_temperature(segment.altitude)
    T0 = atm.get_temperature(0.0)
    nondim_temperature = T / T0
    a = atm.get_speed_of_sound(segment.altitude)

    exponent = -segment.TSFC * g0 / (segment.Mach * math.sqrt(nondim_temperature) * a_standard)
    exponent *= (segment.CD / segment.CL) * segment.Mach * a * segment.delta_t
    segment.weight_fraction = math.exp(exponent)
    return segment


def evaluate_segment(segment: Segment) -> Segment:
    if segment.type == SegmentType.Accelerated_Climb:
        return accelerated_climb(segment)
    elif segment.type == SegmentType.Subsonic_Loiter:
        return subsonic_loiter(segment)
    elif segment.type == SegmentType.Idle:
        return idle_thrust(segment, 1.0)
    elif segment.type == SegmentType.Constant_Altitude_Speed_Cruise:
        return constant_altitude_cruise(segment)
    return segment
