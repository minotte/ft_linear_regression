import pandas as pd
from utils import load_thetas, mean

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

def r_squared(y_true, y_pred):
    """
    Compute the coefficient of determination (R² score).

    The R² score measures how well the predicted values fit the actual data.
    A score close to 1 indicates a good fit, while a score close to 0 means
    the model does not explain the variance of the target variable well.

    Args:
        y_true (iterable): Actual target values.
        y_pred (iterable): Predicted target values.

    Returns:
        float: R² score.
    """
    mean_y = mean(y_true)
    ss_total = sum((y_true - mean_y) ** 2)
    ss_residual = sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_square_error(y_true, y_pred):
    """
    Compute the coefficient of determination (R² score).

    The R² score measures how well the predicted values fit the actual data.
    A score close to 1 indicates a good fit, while a score close to 0 means
    the model does not explain the variance of the target variable well.

    Args:
        y_true (iterable): Actual target values.
        y_pred (iterable): Predicted target values.

    Returns:
        float: R² score.
    """
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

def mean_absolute_error(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    MAE measures the average absolute difference between predicted values
    and actual values. It gives an overall idea of prediction accuracy
    without considering error direction.

    Args:
        y_true (iterable): Actual target values.
        y_pred (iterable): Predicted target values.

    Returns:
        float: Mean absolute error.
    """
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

def root_mean_square_error(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE).

    RMSE is the square root of the Mean Squared Error (MSE).
    It represents the average prediction error in the same unit
    as the target variable.

    Args:
        y_true (iterable): Actual target values.
        y_pred (iterable): Predicted target values.

    Returns:
        float: Root mean squared error.
    """
    return mean_square_error(y_true, y_pred) ** 0.5

def mape(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    MAPE measures the average percentage difference between
    predicted values and actual values.

    Args:
        y_true (iterable): Actual target values.
        y_pred (iterable): Predicted target values.

    Returns:
        float: Mean absolute percentage error in percent.
    """
    return sum(abs(t - p) / t for t , p in zip(y_true, y_pred)) * 100 / len(y_true) 


def colorize(val, threshold, must_be_higher=True):
    """
    Display the value in green if the model's “accuracy” is correct; otherwise, the value will be displayed in red
    args:
        - value to compare
        - comparison threshold
        - must_be_higher if the value should be higher than threshold
    return: 
        the colorized value
    """
    if must_be_higher == True:
        color = GREEN if val >= threshold else RED
    else:
        color = GREEN if val <= threshold else RED
    return f"{color}{val:.4f}{RESET}"

def main():
    """
    load data and thetas
    calcul predictions of price
    call functions to calculate Regression Metrics and print them
    """
    try :
        data = pd.read_csv("data.csv")
        km = data["km"].values
        price = data["price"].values
        theta0, theta1 = load_thetas()
        predictions = theta0 + theta1 * km
        mean_price = mean(price)
        score = r_squared(price, predictions)
        print(f"R² score: {colorize(score, 0.7, True)}")
        mse = mean_square_error(price, predictions)
        print(f"mean square error: {mse:.4f}")

        mae = mean_absolute_error(price, predictions)
        print(f"MAE : {colorize(mae, 0.15 * mean_price, False)}")
        
        rmse = root_mean_square_error(price, predictions)
        print(f"RMAE : {colorize(rmse, 0.20 * mean_price, False)}")

        mape0 = mape(price, predictions)
        print(f"MAPE : {colorize(mape0, 10.0, False)}%")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()