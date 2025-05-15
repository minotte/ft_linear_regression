import numpy as np
import pandas as pd
from utils import load_model


def r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def main():
    data = pd.read_csv("data.csv")
    x = data["km"].values
    y = data["price"].values
    theta0, theta1 = load_model()
    predictions = theta0 + theta1 * x
    score = r_squared(y, predictions)
    print(f"R² score: {score:.4f}")

if __name__ == "__main__":
    main()
