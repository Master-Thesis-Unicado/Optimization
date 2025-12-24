import pandas as pd
import numpy as np

# Read with semicolon delimiter and handle European number format
df = pd.read_csv('Test Cases/Optimized Mission/mission_data_2025-12-23_00-16-05.csv', 
                 sep=';', 
                 decimal=',')

print('='*80)
print('COMPREHENSIVE OPTIMIZED MISSION ANALYSIS')
print('='*80)
print(f'\nColumns available: {list(df.columns)[:20]}...')
print(f'Total rows: {len(df)}')

# Phase separation
climb_df = df[df['Phase'] == 'Climb']
cruise_df = df[df['Phase'] == 'Cruise']
descent_df = df[df['Phase'] == 'Descent']

print(f'\nClimb rows: {len(climb_df)}, Cruise rows: {len(cruise_df)}, Descent rows: {len(descent_df)}')

# ==================== OVERALL MISSION ====================
print('\n' + '='*80)
print('OVERALL MISSION METRICS')
print('='*80)

total_distance = df['Cumulative Distance (km)'].max()
total_time = df['Cumulative Time (s)'].max() / 60  # minutes
total_fuel = df['Total Fuel Consumed (kg)'].max()
initial_mass = df['Mass (kg)'].iloc[0]
final_mass = df['Mass (kg)'].iloc[-1]

print(f'Total Distance: {total_distance:.1f} km')
print(f'Total Duration: {total_time:.1f} min ({total_time/60:.2f} hours)')
print(f'Total Fuel Consumed: {total_fuel:.1f} kg')
print(f'Initial Mass: {initial_mass:.0f} kg')
print(f'Final Mass: {final_mass:.0f} kg')
print(f'Fuel Efficiency: {total_fuel/total_distance:.2f} kg/km')

# ==================== CLIMB PHASE ====================
print('\n' + '='*80)
print('CLIMB PHASE METRICS')
print('='*80)

climb_duration = (climb_df['Cumulative Time (s)'].max() - climb_df['Cumulative Time (s)'].min()) / 60
climb_distance = climb_df['Cumulative Distance (km)'].max() - climb_df['Cumulative Distance (km)'].min()
climb_fuel = climb_df['Total Fuel Consumed (kg)'].iloc[-1] - climb_df['Total Fuel Consumed (kg)'].iloc[0]
climb_alt_gain = climb_df['Altitude (m)'].max() - climb_df['Altitude (m)'].min()

print(f'Climb Duration: {climb_duration:.1f} min')
print(f'Climb Distance: {climb_distance:.1f} km')
print(f'Climb Fuel: {climb_fuel:.1f} kg')
print(f'Altitude Gain: {climb_alt_gain:.0f} m')
print(f'Average Climb Rate: {climb_alt_gain/climb_duration:.0f} m/min')

# Fuel flow
print(f'\nFuel Flow:')
print(f'  Initial: {climb_df["Fuel Flow (kg/h)"].iloc[0]:.0f} kg/h')
print(f'  Max: {climb_df["Fuel Flow (kg/h)"].max():.0f} kg/h')
print(f'  Min: {climb_df["Fuel Flow (kg/h)"].min():.0f} kg/h')
print(f'  Average: {climb_df["Fuel Flow (kg/h)"].mean():.0f} kg/h')

# Thrust and Drag
print(f'\nThrust:')
print(f'  Initial: {climb_df["Thrust Total (N)"].iloc[0]/1000:.0f} kN')
print(f'  Max: {climb_df["Thrust Total (N)"].max()/1000:.0f} kN')
print(f'  Min: {climb_df["Thrust Total (N)"].min()/1000:.0f} kN')
print(f'  Final: {climb_df["Thrust Total (N)"].iloc[-1]/1000:.0f} kN')

print(f'\nDrag:')
print(f'  Initial: {climb_df["Drag (N)"].iloc[0]/1000:.0f} kN')
print(f'  Max: {climb_df["Drag (N)"].max()/1000:.0f} kN')
print(f'  Min: {climb_df["Drag (N)"].min()/1000:.0f} kN')
print(f'  Final: {climb_df["Drag (N)"].iloc[-1]/1000:.0f} kN')

# Velocity and Mach
print(f'\nTrue Airspeed:')
print(f'  Initial: {climb_df["True Airspeed (m/s)"].iloc[0]:.0f} m/s')
print(f'  Final: {climb_df["True Airspeed (m/s)"].iloc[-1]:.0f} m/s')

print(f'\nMach Number:')
print(f'  Initial: {climb_df["Mach Number"].iloc[0]:.2f}')
print(f'  Final: {climb_df["Mach Number"].iloc[-1]:.2f}')

# Throttle (Lever Position)
print(f'\nThrottle (Lever Position):')
print(f'  Initial: {climb_df["Lever Position"].iloc[0]*100:.0f}%')
print(f'  Max: {climb_df["Lever Position"].max()*100:.0f}%')
print(f'  Min: {climb_df["Lever Position"].min()*100:.0f}%')
print(f'  Final: {climb_df["Lever Position"].iloc[-1]*100:.0f}%')

# Mass
print(f'\nMass:')
print(f'  Initial: {climb_df["Mass (kg)"].iloc[0]:.0f} kg')
print(f'  Final: {climb_df["Mass (kg)"].iloc[-1]:.0f} kg')

# ==================== CRUISE PHASE ====================
print('\n' + '='*80)
print('CRUISE PHASE METRICS')
print('='*80)

cruise_duration = (cruise_df['Cumulative Time (s)'].max() - cruise_df['Cumulative Time (s)'].min()) / 60
cruise_distance = cruise_df['Cumulative Distance (km)'].max() - cruise_df['Cumulative Distance (km)'].min()
cruise_fuel = cruise_df['Total Fuel Consumed (kg)'].iloc[-1] - cruise_df['Total Fuel Consumed (kg)'].iloc[0]

print(f'Cruise Duration: {cruise_duration:.1f} min ({cruise_duration/60:.2f} hours)')
print(f'Cruise Distance: {cruise_distance:.1f} km')
print(f'Cruise Fuel: {cruise_fuel:.1f} kg')
print(f'Cruise Altitude: {cruise_df["Altitude (m)"].mean():.0f} m')

print(f'\nFuel Flow:')
print(f'  Initial: {cruise_df["Fuel Flow (kg/h)"].iloc[0]:.0f} kg/h')
print(f'  Final: {cruise_df["Fuel Flow (kg/h)"].iloc[-1]:.0f} kg/h')
print(f'  Average: {cruise_df["Fuel Flow (kg/h)"].mean():.0f} kg/h')

print(f'\nThrust and Drag:')
print(f'  Thrust Initial: {cruise_df["Thrust Total (N)"].iloc[0]/1000:.1f} kN')
print(f'  Thrust Final: {cruise_df["Thrust Total (N)"].iloc[-1]/1000:.1f} kN')
print(f'  Drag Initial: {cruise_df["Drag (N)"].iloc[0]/1000:.1f} kN')
print(f'  Drag Final: {cruise_df["Drag (N)"].iloc[-1]/1000:.1f} kN')

print(f'\nVelocity:')
print(f'  True Airspeed: {cruise_df["True Airspeed (m/s)"].mean():.1f} m/s')
print(f'  Mach Number: {cruise_df["Mach Number"].mean():.2f}')

print(f'\nThrottle:')
print(f'  Initial: {cruise_df["Lever Position"].iloc[0]*100:.0f}%')
print(f'  Final: {cruise_df["Lever Position"].iloc[-1]*100:.0f}%')
print(f'  Average: {cruise_df["Lever Position"].mean()*100:.0f}%')

print(f'\nMass:')
print(f'  Initial: {cruise_df["Mass (kg)"].iloc[0]:.0f} kg')
print(f'  Final: {cruise_df["Mass (kg)"].iloc[-1]:.0f} kg')

# Specific Range
specific_range = cruise_distance * 1000 / cruise_fuel  # m/kg
print(f'  Specific Range: {specific_range:.1f} m/kg')

# ==================== DESCENT PHASE ====================
print('\n' + '='*80)
print('DESCENT PHASE METRICS')
print('='*80)

descent_duration = (descent_df['Cumulative Time (s)'].max() - descent_df['Cumulative Time (s)'].min()) / 60
descent_distance = descent_df['Cumulative Distance (km)'].max() - descent_df['Cumulative Distance (km)'].min()
descent_fuel = descent_df['Total Fuel Consumed (kg)'].iloc[-1] - descent_df['Total Fuel Consumed (kg)'].iloc[0]
descent_alt_loss = descent_df['Altitude (m)'].iloc[0] - descent_df['Altitude (m)'].iloc[-1]

print(f'Descent Duration: {descent_duration:.1f} min')
print(f'Descent Distance: {descent_distance:.1f} km')
print(f'Descent Fuel: {descent_fuel:.1f} kg')
print(f'Altitude Loss: {descent_alt_loss:.0f} m')
print(f'Average Descent Rate: {descent_alt_loss/descent_duration:.0f} m/min')

print(f'\nFuel Flow:')
print(f'  Initial: {descent_df["Fuel Flow (kg/h)"].iloc[0]:.0f} kg/h')
print(f'  Max: {descent_df["Fuel Flow (kg/h)"].max():.0f} kg/h')
print(f'  Final: {descent_df["Fuel Flow (kg/h)"].iloc[-1]:.0f} kg/h')
print(f'  Average: {descent_df["Fuel Flow (kg/h)"].mean():.0f} kg/h')

print(f'\nThrust:')
print(f'  Initial: {descent_df["Thrust Total (N)"].iloc[0]/1000:.0f} kN')
print(f'  Max: {descent_df["Thrust Total (N)"].max()/1000:.0f} kN')
print(f'  Final: {descent_df["Thrust Total (N)"].iloc[-1]/1000:.0f} kN')

print(f'\nDrag:')
print(f'  Initial: {descent_df["Drag (N)"].iloc[0]/1000:.0f} kN')
print(f'  Max: {descent_df["Drag (N)"].max()/1000:.0f} kN')
print(f'  Final: {descent_df["Drag (N)"].iloc[-1]/1000:.0f} kN')

print(f'\nVelocity:')
print(f'  Initial TAS: {descent_df["True Airspeed (m/s)"].iloc[0]:.0f} m/s')
print(f'  Max TAS: {descent_df["True Airspeed (m/s)"].max():.0f} m/s')
print(f'  Final TAS: {descent_df["True Airspeed (m/s)"].iloc[-1]:.0f} m/s')

print(f'\nMach Number:')
print(f'  Initial: {descent_df["Mach Number"].iloc[0]:.2f}')
print(f'  Max: {descent_df["Mach Number"].max():.2f}')
print(f'  Final: {descent_df["Mach Number"].iloc[-1]:.2f}')

print(f'\nThrottle:')
print(f'  Value: {descent_df["Lever Position"].mean()*100:.0f}%')

print(f'\nMass:')
print(f'  Initial: {descent_df["Mass (kg)"].iloc[0]:.0f} kg')
print(f'  Final: {descent_df["Mass (kg)"].iloc[-1]:.0f} kg')

# ==================== L/D and CD ====================
print('\n' + '='*80)
print('AERODYNAMIC METRICS')
print('='*80)

print(f'\nClimb Phase:')
print(f'  Max L/D: {climb_df["L/D Ratio"].max():.1f}')
print(f'  Avg L/D: {climb_df["L/D Ratio"].mean():.1f}')
print(f'  Avg CD: {climb_df["Drag Coefficient"].mean():.4f}')

print(f'\nCruise Phase:')
print(f'  Avg L/D: {cruise_df["L/D Ratio"].mean():.1f}')
print(f'  Avg CD: {cruise_df["Drag Coefficient"].mean():.4f}')

print(f'\nDescent Phase:')
print(f'  Avg L/D: {descent_df["L/D Ratio"].mean():.1f}')
print(f'  Avg CD: {descent_df["Drag Coefficient"].mean():.4f}')

