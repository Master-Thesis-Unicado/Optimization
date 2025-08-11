from climb import generate_strategy, simulate_climb_path, target_altitude
from plotting import interactive_plot

def main():
    print("Starting full simulation ...")
    interactive_plot(
        generate_strategy_func=generate_strategy,
        simulate_func=simulate_climb_path,
        target_altitude=target_altitude
    )

if __name__ == "__main__":
    main()
