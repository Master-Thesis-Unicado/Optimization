import os
from pyaerodynamics import Aircraft, Flight_Condition

file_path = os.path.join("polar.xml")

# Create the Aircraft from XML
aircraft = Aircraft(file_path)

# Define flight conditions
conditions = Flight_Condition(4000, 0.6)

# Change settings
aircraft.change_settings("main_wing", "horizontal_stabiliser", "linear")

# Center of gravity location
cog_location = [14.0, 0.0, 0.0]

# Run linearized trim
aircraft.linearized_trim(conditions, 600000.0, cog_location)

# Extract results
wing_aoa = aircraft.settings.reference_wing_angle
ht_incidence = aircraft.settings.adjustable_surface_angle

print(f"wing angle of attack = {wing_aoa} deg")
print(f"ht incidence = {ht_incidence} deg")

cd = aircraft.get_CD(conditions, 600000.0)
ld = aircraft.get_CL_CD(conditions, 600000.0, cog_location)

print(f"aircraft CD = {cd}")
print(f"aircraft L/D = {ld}")