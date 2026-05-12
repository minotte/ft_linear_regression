import pandas as pd

def load_data(filename="data.csv"):
    """
    This function loads mileage and price data from a CSV file.

    Args:
        - filename(str): the name of the CSV, should containing 'km' and 'price' columns (default: 'data.csv').

    Returns:
        - mileage, price

    Error:
        - Prints an error and exits if the file is missing or malformed.
    """
    try:
        data = pd.read_csv(filename)
        km = data["km"].values
        price = data["price"].values
        return km, price
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def load_thetas(filename="model.txt"):
    """
    This function loads thetas data from the model file.

    Args:
        - filename(str): the name of the model file, should containing 'theta0' and 'theta1' (default: 'model').

    Returns:
        - theta0, theta1

    Error:
        - Prints an error and return the default thetas : theta0 = 0 theta1 = 0.
    """
    try:
        with open(filename, "r") as f:
            theta0, theta1 = map(float, f.read().strip().split(","))
            return theta0, theta1
    except FileNotFoundError:
        return 0.0, 0.0

def save_model(theta0:float, theta1:float, filename="model.txt"):
    """
    Save theta0, theta1 in a file
    Args:
        - theta0 and theta1 : float
        - filename: name str (default: 'model').
    Error:
        - Prints an error and exit
    """
    try:
        with open(filename, "w") as f:
            f.write(f"{theta0},{theta1}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def mean(data) -> float:
    """
    Calculates the mean (average) of a list of numbers.

    Args:
        data: List of numbers.

    Returns:
        float: The mean of the list.
    """
    return (sum(data) / len(data))

def variance(data: list) -> float:
    """
    Calculates the variance of a list of numbers.

    Args:
        data: List of numbers.

    Returns:
        float: The variance of the list.
    """
    m = mean(data)
    var = sum((x - m) ** 2 for x in data) / len(data)
    return var

def standard_deviation(data: list) -> float:
    """
    Calculates the standard deviation of a list of numbers.

    Args:
        data : List of numbers.

    Returns:
        float: The standard deviation of the list.
    """
    var = variance(data)
    std = var ** 0.5
    return std