import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StandardScaler:
    """
    A class to standardize (normalize) and revert normalization for 1D linear data.

    Purpose:
        - Normalize input features (e.g., mileage) and target values (e.g., price)
          by removing the mean and scaling to unit variance.
        - Revert model coefficients learned on normalized data back to the original scale.

    Methods:
        - __init__(x, y): Initializes with original x and y, and computes means and std deviations.
        - normalize(x, y): Returns normalized versions of x and y.
        - revert_coefficients(norm_theta0, norm_theta1): Reverts normalized coefficients
          to match the original data scale.

    Args:
        - x (np.array): input feature (e.g., mileage).
        - y (np.array): target values (e.g., price).

    Stored attributes:
        - mean_x (float): mean of input x.
        - std_x (float): standard deviation of x.
        - mean_y (float): mean of target y.
        - std_y (float): standard deviation of y.
    """
    def __init__(self, x, y):
        self.mean_x = np.mean(x)
        self.std_x = np.std(x)
        self.mean_y = np.mean(y)
        self.std_y = np.std(y)

    def normalize(self, x, y):
        norm_x = (x - self.mean_x) / self.std_x
        norm_y = (y - self.mean_y) / self.std_y
        return norm_x, norm_y

    def revert_coefficients(self, norm_theta0, norm_theta1):
        theta1 = norm_theta1 * self.std_y / self.std_x
        theta0 = self.mean_y - theta1 * self.mean_x + norm_theta0 * self.std_y
        return theta0, theta1


def load_data(filename="data.csv"):
    """
    This function loads mileage and price data from a CSV file.
    
    Args:
        - filename (str): the name of the CSV file containing 'km' and 'price' columns (default: 'data.csv').
    
    Returns:
        - Tuple of numpy arrays: (mileage, price)
    
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


def gradient_descent(x, y, learning_rate=0.57, iterations=1000):
    """
    This function performs gradient descent to fit a line on normalized data using MSE.
    
    Args:
        - x (np.array): normalized input feature (e.g., mileage).
        - y (np.array): normalized target values (e.g., price).
        - learning_rate (float): the learning rate for updating theta (default: 0.1).
        - iterations (int): number of iterations for gradient descent (default: 1000).
    
    Returns:
        - Tuple: (normalized_theta0, normalized_theta1)
    
    Purpose:
        To minimize the error between predictions and true values using gradient descent.
    """
    theta0, theta1 = 0.0, 0.0
    m = len(x)

    for i in range(iterations):
        predictions = theta0 + theta1 * x
        error = predictions - y
        theta0 -= learning_rate * (1 / m) * np.sum(error)
        theta1 -= learning_rate * (1 / m) * np.sum(error * x)

    return theta0, theta1


def save_model(theta0, theta1, filename="model.txt"):
    """
    This function saves the learned model parameters to a file.
    
    Args:
        - theta0 (float): intercept of the model.
        - theta1 (float): slope of the model.
        - filename (str): output filename to store the model (default: 'model.txt').
    
    File content:
        A single line with two comma-separated values: theta0,theta1
    """
    with open(filename, "w") as f:
        f.write(f"{theta0},{theta1}")

def show_graph(mileage, price, theta0, theta1):
    plt.scatter(mileage, price, color="blue", label="Data")
    x_line = np.linspace(min(mileage), max(mileage), 100)
    y_line = theta0 + theta1 * x_line
    plt.plot(x_line, y_line, color="red", label="linear regression")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.title("Simple linear regression")
    plt.show()

def main():
    """
    It's the main function of the program. It loads the data, normalizes it,
    trains a linear regression model using gradient descent, denormalizes
    the model parameters, saves them, and visualizes the result.
    
    Steps:
        - Load mileage and price data from CSV.
        - Normalize the values using StandardScaler.
        - Run gradient descent on normalized data.
        - Revert the normalized model to original scale.
        - Save the final theta0 and theta1.
        - Plot the regression line over real data.
    
    Print:
        - Final model in the format: price = theta0 + theta1 * km
    """
    try:
        mileage, price = load_data()

        scaler = StandardScaler(mileage, price)
        mileage_norm, y_norm = scaler.normalize(mileage, price)

        norm_theta0, norm_theta1 = gradient_descent(mileage_norm, y_norm, learning_rate=0.1, iterations=1000)
        theta0, theta1 = scaler.revert_coefficients(norm_theta0, norm_theta1)
        save_model(theta0, theta1)
        show_graph(mileage, price, theta0, theta1)

    except KeyboardInterrupt:
        print("\nGood bye!")

if __name__ == "__main__":
    main()
 