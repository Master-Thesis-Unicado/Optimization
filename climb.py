import numpy as np
from atmosphere import Atmosphere

# Initialize atmospheric model
atm = Atmosphere()

# Constants
INITIAL_SPEED = 75
INITIAL_ALTITUDE = 0
TARGET_ALTITUDE = 10000
E_DOT = 10.0
altitude_fractions = np.linspace(0.1, 0.9, 15)

class StrategyProfiles:
    class FixedEnergy:
        class Linear:
            @staticmethod
            def profile(altitude, velocity, altitude_fraction):
                g = atm.get_gravity(altitude)
                dh_dt = altitude_fraction * E_DOT
                dv_dt = (1.0 - altitude_fraction) * E_DOT * g / velocity
                return dh_dt, dv_dt

        class Exponential:
            @staticmethod
            def increasing_climb(altitude, velocity, altitude_fraction):
                g = atm.get_gravity(altitude)
                climb_weight = altitude_fraction * np.exp(altitude / TARGET_ALTITUDE)
                speed_weight = (1.0 - altitude_fraction) * np.exp(-altitude / TARGET_ALTITUDE)
                total = climb_weight + speed_weight
                climb_fraction = climb_weight / total
                speed_fraction = speed_weight / total
                return climb_fraction * E_DOT, speed_fraction * E_DOT * g / velocity

            @staticmethod
            def decreasing_climb(altitude, velocity, altitude_fraction):
                g = atm.get_gravity(altitude)
                climb_weight = altitude_fraction * np.exp(-altitude / TARGET_ALTITUDE)
                speed_weight = (1.0 - altitude_fraction) * np.exp(altitude / TARGET_ALTITUDE)
                total = climb_weight + speed_weight
                climb_fraction = climb_weight / total
                speed_fraction = speed_weight / total
                return climb_fraction * E_DOT, speed_fraction * E_DOT * g / velocity

            @staticmethod
            def increasing_speed(altitude, velocity, altitude_fraction):
                g = atm.get_gravity(altitude)
                speed_weight = altitude_fraction * np.exp(altitude / TARGET_ALTITUDE)
                climb_weight = (1.0 - altitude_fraction) * np.exp(-altitude / TARGET_ALTITUDE)
                total = climb_weight + speed_weight
                climb_fraction = climb_weight / total
                speed_fraction = speed_weight / total
                return climb_fraction * E_DOT, speed_fraction * E_DOT * g / velocity

            @staticmethod
            def decreasing_speed(altitude, velocity, altitude_fraction):
                g = atm.get_gravity(altitude)
                speed_weight = altitude_fraction * np.exp(-altitude / TARGET_ALTITUDE)
                climb_weight = (1.0 - altitude_fraction) * np.exp(altitude / TARGET_ALTITUDE)
                total = climb_weight + speed_weight
                climb_fraction = climb_weight / total
                speed_fraction = speed_weight / total
                return climb_fraction * E_DOT, speed_fraction * E_DOT * g / velocity

    class ConstantRates:
        @staticmethod
        def constant_speed(altitude, velocity, altitude_fraction=None):
            g = atm.get_gravity(altitude)
            return g, 0.0


def generate_strategy(profile='linear'):
    strategies = []
    for af in altitude_fractions:
        if profile == 'linear':
            func = StrategyProfiles.FixedEnergy.Linear.profile
            strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))
        elif profile == 'exponential_increasing_climb':
            func = StrategyProfiles.FixedEnergy.Exponential.increasing_climb
            strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))
        elif profile == 'exponential_decreasing_climb':
            func = StrategyProfiles.FixedEnergy.Exponential.decreasing_climb
            strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))
        elif profile == 'exponential_increasing_speed':
            func = StrategyProfiles.FixedEnergy.Exponential.increasing_speed
            strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))
        elif profile == 'exponential_decreasing_speed':
            func = StrategyProfiles.FixedEnergy.Exponential.decreasing_speed
            strategies.append((af, lambda h, V, af=af, f=func: f(h, V, af)))

    # Add only one strategy for constant speed (not dependent on altitude_fraction)
    if profile == 'constant_speed':
        func = StrategyProfiles.ConstantRates.constant_speed
        strategies.append((None, func))

    return strategies


def simulate_climb_path(strategy_function, dt=1.0):
    h, V, t = [INITIAL_ALTITUDE], [INITIAL_SPEED], [0]

    while h[-1] < TARGET_ALTITUDE:
        altitude, velocity = h[-1], V[-1]
        dh_dt, dv_dt = strategy_function(altitude, velocity)

        h_new = altitude + dh_dt * dt
        V_new = velocity + dv_dt * dt

        if h_new >= TARGET_ALTITUDE:
            h_new = TARGET_ALTITUDE
            t_new = t[-1] + (TARGET_ALTITUDE - altitude) / max(dh_dt, 1e-3)
            V_new = velocity + dv_dt * (t_new - t[-1])
            t.append(t_new)
            h.append(h_new)
            V.append(V_new)
            break

        h.append(h_new)
        V.append(V_new)
        t.append(t[-1] + dt)

    return t, h, V
