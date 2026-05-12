import pandas as pd
import sys
from utils import load_thetas


def     estimation(mileage: float, theta0:float, theta1:float):
    """
    This function will calculate the estimation and return it

    Args:
        - mileage of car you want estimate the price
        - theta0: by default 0
        - theta1: by default 0

    Returns:
        - estimation of the price
    """
    return (theta0 + (theta1 * mileage))

def main():
    try:
        mileage = float(input("enter the mileage:"))
        assert mileage > 0, "should be positive"
        theta0, theta1 = load_thetas()
        estimationPrice = estimation(mileage,theta0, theta1)
        print(f"Estimated price: {estimationPrice:.2f}")
    except Exception as e:
        print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nInterruption... Bye!")
        sys.exit(130)

if __name__ == "__main__":
    main()
