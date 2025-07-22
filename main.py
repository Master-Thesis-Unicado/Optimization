from mission_state import MissionState
from atmosphere import Atmosphere
from climb import run_climb
from cruise import run_cruise
from descent import run_descent

def main():
    state = MissionState()
    atm = Atmosphere()

    state = run_climb(state, atm)
    state = run_cruise(state, atm)
    state = run_descent(state, atm)

    print(state)

if __name__ == "__main__":
    main()
