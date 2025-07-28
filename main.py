from climb import generate_strategy, simulate_climb_path, TARGET_ALTITUDE
from plotting import interactive_plot

def main():
    interactive_plot(
        generate_strategy_func=generate_strategy,
        simulate_func=simulate_climb_path,
        target_altitude=TARGET_ALTITUDE
    )

if __name__ == "__main__":
    main()
