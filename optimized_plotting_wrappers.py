"""
Optimized Mission Plotting Wrappers

This module provides a context manager to redirect all plotting outputs from 
main_optimized.py to the 'Optimized' subfolder instead of the standard 
phase-specific folders (Climb, Cruise, Descent).

This ensures that all outputs from the fuel optimization workflow are organized
together in: Images/YYYY-MM-DD_HH-MM-SS/Optimized/

Scientific context: Maintaining organized output structure facilitates
comparative analysis between standard mission profiles and fuel-optimized
trajectories, enabling systematic evaluation of optimization effectiveness.

Usage:
    with OptimizedOutputRedirect():
        # All plotting functions will save to Optimized folder
        plot_J_3d_plotly(...)
        plot_cruise_performance_detailed(...)
        # etc.
"""

from __future__ import annotations
import visualization_config
from contextlib import contextmanager


@contextmanager
def OptimizedOutputRedirect():
    """
    Context manager that redirects all plotting outputs to the 'Optimized' subfolder.
    
    This works by temporarily patching the get_or_create_run_directory function
    to always return the Optimized folder path, regardless of which phase is requested.
    
    Usage:
        with OptimizedOutputRedirect():
            plot_J_3d_plotly(...)              # Saves to Optimized/ instead of Climb/
            plot_cruise_performance_detailed(...)  # Saves to Optimized/ instead of Cruise/
            plot_descent_trajectory_interactive(...)  # Saves to Optimized/ instead of Descent/
            # etc.
    
    Example:
        # Standard mission analysis - outputs go to Climb/, Cruise/, Descent/
        plot_J_3d_plotly(...)
        
        # Optimized mission analysis - all outputs go to Optimized/
        with OptimizedOutputRedirect():
            plot_J_3d_plotly(...)
            plot_mission_summary_dashboard(...)
    """
    # Store the original function
    original_get_dir = visualization_config.get_or_create_run_directory
    
    # Create wrapper that always returns Optimized folder
    def optimized_get_dir(base_dir=None, phase=None):
        # Always use "Optimized" as the phase, ignoring what was requested
        return original_get_dir(base_dir=base_dir, phase="Optimized")
    
    # Temporarily replace the function
    visualization_config.get_or_create_run_directory = optimized_get_dir
    
    try:
        yield  # Let the context block execute
    finally:
        # Restore the original function
        visualization_config.get_or_create_run_directory = original_get_dir

