# Fuel Plotting Module - Mathematical and Logical Issues Review

## Summary
This document identifies mathematical and logical errors found in `fuel_plotting.py`, focusing on division by zero, unsafe array indexing, and incorrect calculations.

---

## Critical Mathematical Errors

### 1. **Division by Zero - Line 72**
**Location:** `plot_convergence_trajectory()`

**Issue:**
```python
first_fuel = consumed_fuels[0]
fuel_percent_change = [(f - first_fuel) / first_fuel * 100 for f in consumed_fuels]
```

**Problem:** If the first iteration consumed 0 kg of fuel, this causes `ZeroDivisionError`.

**Recommendation:**
```python
first_fuel = consumed_fuels[0]
if first_fuel > 0:
    fuel_percent_change = [(f - first_fuel) / first_fuel * 100 for f in consumed_fuels]
else:
    fuel_percent_change = [0.0] * len(consumed_fuels)
```

---

### 2. **Division by Zero - Line 391**
**Location:** `plot_kpp_evolution()`

**Issue:**
```python
fuel_savings = first_result.initial_fuel_kg - optimized_capacity
percent_savings = (fuel_savings / first_result.initial_fuel_kg) * 100
```

**Problem:** If `first_result.initial_fuel_kg == 0`, this causes `ZeroDivisionError`.

**Recommendation:**
```python
if first_result.initial_fuel_kg > 0:
    fuel_savings = first_result.initial_fuel_kg - optimized_capacity
    percent_savings = (fuel_savings / first_result.initial_fuel_kg) * 100
else:
    fuel_savings = 0.0
    percent_savings = 0.0
```

---

### 3. **Division by Zero - Lines 395-397**
**Location:** `plot_kpp_evolution()`

**Issue:**
```python
total_fuel = last_result.fuel_consumed_kg
climb_pct = (last_result.climb_fuel_kg / total_fuel) * 100
cruise_pct = (last_result.cruise_fuel_kg / total_fuel) * 100
descent_pct = (last_result.descent_fuel_kg / total_fuel) * 100
```

**Problem:** If `total_fuel == 0`, all three calculations cause `ZeroDivisionError`.

**Recommendation:**
```python
total_fuel = last_result.fuel_consumed_kg
if total_fuel > 0:
    climb_pct = (last_result.climb_fuel_kg / total_fuel) * 100
    cruise_pct = (last_result.cruise_fuel_kg / total_fuel) * 100
    descent_pct = (last_result.descent_fuel_kg / total_fuel) * 100
else:
    climb_pct = cruise_pct = descent_pct = 0.0
```

---

### 4. **Division by Zero - Line 400**
**Location:** `plot_kpp_evolution()`

**Issue:**
```python
avg_speed_kmh = last_result.total_distance_km / (last_result.total_time_s / 3600)
```

**Problem:** If `total_time_s == 0`, this causes `ZeroDivisionError`.

**Recommendation:**
```python
if last_result.total_time_s > 0:
    avg_speed_kmh = last_result.total_distance_km / (last_result.total_time_s / 3600)
else:
    avg_speed_kmh = 0.0
```

---

### 5. **Division by Zero - Line 401**
**Location:** `plot_kpp_evolution()`

**Issue:**
```python
fuel_efficiency = last_result.fuel_consumed_kg / last_result.total_distance_km
```

**Note:** While protected by the check at line 248-251 for the loop, this standalone calculation at line 401 is NOT protected. However, if `total_distance_km == 0`, this will cause `ZeroDivisionError`.

**Recommendation:**
```python
if last_result.total_distance_km > 0:
    fuel_efficiency = last_result.fuel_consumed_kg / last_result.total_distance_km
else:
    fuel_efficiency = 0.0
```

---

### 6. **Division by Zero - Lines 529-531**
**Location:** `plot_optimization_comparison()`

**Issue:**
```python
fuel_reduction_pct = (fuel_reduction / first.fuel_consumed_kg) * 100
mass_reduction = first.initial_mass_kg - final.initial_mass_kg
mass_reduction_pct = (mass_reduction / first.initial_mass_kg) * 100
```

**Problem:** If `first.fuel_consumed_kg == 0` or `first.initial_mass_kg == 0`, these cause `ZeroDivisionError`.

**Recommendation:**
```python
fuel_reduction = first.fuel_consumed_kg - final.fuel_consumed_kg
fuel_reduction_pct = (fuel_reduction / first.fuel_consumed_kg * 100) if first.fuel_consumed_kg > 0 else 0.0

mass_reduction = first.initial_mass_kg - final.initial_mass_kg
mass_reduction_pct = (mass_reduction / first.initial_mass_kg * 100) if first.initial_mass_kg > 0 else 0.0
```

---

### 7. **Division by Zero - Lines 533-535**
**Location:** `plot_optimization_comparison()`

**Issue:**
```python
first_efficiency = first.fuel_consumed_kg / first.total_distance_km
final_efficiency = final.fuel_consumed_kg / final.total_distance_km
efficiency_improvement = ((first_efficiency - final_efficiency) / first_efficiency) * 100
```

**Problems:** 
- If `first.total_distance_km == 0` or `final.total_distance_km == 0`, lines 533-534 cause `ZeroDivisionError`.
- If `first_efficiency == 0`, line 535 causes `ZeroDivisionError`.

**Recommendation:**
```python
first_efficiency = (first.fuel_consumed_kg / first.total_distance_km) if first.total_distance_km > 0 else 0.0
final_efficiency = (final.fuel_consumed_kg / final.total_distance_km) if final.total_distance_km > 0 else 0.0

if first_efficiency > 0:
    efficiency_improvement = ((first_efficiency - final_efficiency) / first_efficiency) * 100
else:
    efficiency_improvement = 0.0
```

---

## Critical Array Indexing Errors

### 8. **Unsafe Array Indexing - Lines 798, 811, 831, 845, 859, 879**
**Location:** `plot_3d_trajectory_comparison()`

**Issue:**
Multiple unsafe accesses to `[-1]` elements without checking if arrays are empty:

```python
# Line 798
climb_dist_first = np.linspace(0, 50, len(first.climb_result.alt_m))
climb_time_first = np.cumsum(first.climb_result.dt_s) / 60  # Could be empty

# Line 811
cruise_dist_first = np.linspace(climb_dist_first[-1], climb_dist_first[-1] + first.cruise_result.distance_km[-1], 
                               len(first.cruise_result.distance_km))

# Line 831
descent_time_first = cruise_time_first[-1] + np.cumsum(first.descent_result.dt_s) / 60
```

**Problem:** If arrays are empty, `[-1]` raises `IndexError`. Also, `np.cumsum([])` returns an empty array, and accessing `[-1]` on an empty array fails.

**Recommendation:**
```python
# Check before accessing
if len(first.climb_result.alt_m) > 0 and len(first.climb_result.dt_s) > 0:
    climb_dist_first = np.linspace(0, 50, len(first.climb_result.alt_m))
    climb_time_first = np.cumsum(first.climb_result.dt_s) / 60
    # ... continue with trajectory
else:
    # Handle empty arrays gracefully
    return  # or skip plotting this phase
```

**Same issue exists for:**
- Line 845-846 (final iteration climb)
- Lines 811-813 (first iteration cruise)
- Lines 859-861 (final iteration cruise)
- Lines 831 (first iteration descent)
- Lines 879 (final iteration descent)

---

### 9. **Unsafe Array Indexing - Lines 811, 859**
**Location:** `plot_3d_trajectory_comparison()`

**Issue:**
```python
# Line 811
cruise_dist_first = np.linspace(climb_dist_first[-1], climb_dist_first[-1] + first.cruise_result.distance_km[-1], 
                               len(first.cruise_result.distance_km))
```

**Problem:** 
- `climb_dist_first[-1]` could fail if `climb_dist_first` is empty
- `first.cruise_result.distance_km[-1]` could fail if `distance_km` array is empty

**Recommendation:**
```python
if (len(climb_dist_first) > 0 and 
    len(first.cruise_result.distance_km) > 0):
    cruise_dist_first = np.linspace(
        climb_dist_first[-1], 
        climb_dist_first[-1] + first.cruise_result.distance_km[-1], 
        len(first.cruise_result.distance_km)
    )
```

---

## Logical Issues

### 10. **Inconsistent Fuel Savings Calculation - Lines 63-66 vs 390-391**
**Location:** Two different functions calculate fuel savings differently

**Issue 1 (Line 63-66):**
```python
original_max_fuel = 23860.0  # A320 maximum fuel capacity in kg
optimized_fuel = last_iter.fuel_consumed_kg * (1.0 + SAFETY_BUFFER_PERCENT)
fuel_savings = original_max_fuel - optimized_fuel
```

**Issue 2 (Line 390-391):**
```python
optimized_capacity = last_result.fuel_consumed_kg * (1 + SAFETY_BUFFER_PERCENT)
fuel_savings = first_result.initial_fuel_kg - optimized_capacity
```

**Problem:** 
- First function compares against A320 maximum capacity (hardcoded value)
- Second function compares against first iteration's initial fuel estimate
- These are fundamentally different comparisons and may yield different results

**Recommendation:** Decide on a consistent baseline. If comparing optimization effectiveness, use `first_result.initial_fuel_kg`. If comparing against aircraft maximum, use A320 capacity but document why.

---

### 11. **Potential Negative Fuel Savings**
**Location:** Lines 65, 390, 526

**Issue:**
If `optimized_fuel > original_max_fuel` (line 65) or `optimized_capacity > first_result.initial_fuel_kg` (line 390), fuel_savings becomes negative. This may be intentional (showing a penalty), but should be documented.

**Recommendation:** Add validation and appropriate handling/documentation for negative savings cases.

---

### 12. **Hardcoded Gravity Constant - Line 939-941**
**Location:** `plot_specific_energy_evolution()`

**Issue:**
```python
eh_climb = [r.avg_specific_energy_climb_J_kg / 9.81 for r in history.iterations]
```

**Problem:** Hardcoded `9.81` instead of using `G_C` from aircraft_config. While numerically correct, it's inconsistent with the rest of the codebase.

**Recommendation:**
```python
from aircraft_config import G_C
eh_climb = [r.avg_specific_energy_climb_J_kg / G_C for r in history.iterations]
```

---

## Summary of Required Fixes

### Priority 1 (Critical - Can cause crashes):
1. Add zero-division protection for lines 72, 391, 395-397, 400, 401, 529-531, 533-535
2. Add array bounds checking before `[-1]` accesses in `plot_3d_trajectory_comparison()`

### Priority 2 (Logic errors):
3. Standardize fuel savings calculation baseline (A320 max vs first iteration initial fuel)
4. Replace hardcoded gravity with `G_C` constant

### Priority 3 (Code quality):
5. Document handling of negative fuel savings
6. Add validation for empty arrays earlier in functions

---

## Testing Recommendations

Test edge cases:
1. Zero fuel consumption in first iteration
2. Zero total distance
3. Zero total time
4. Empty arrays for altitude, time, distance
5. Zero phase fuel (climb/cruise/descent)
6. Negative fuel savings scenarios
