import sys
import matplotlib.pyplot as plt
from utils import mean, standard_deviation, load_data, save_model, load_thetas
from prediction import estimation

def show_graph(mileage, price, price_pred):
    plt.figure(figsize=(8,6)) 
    plt.scatter(mileage, price, color='purple', label='Data Points') 
    plt.plot(mileage, price_pred, color='red', linewidth=2, label='Regression Line') 
    plt.title('Linear Regression on Random Dataset')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.legend()
    plt.grid(True)
    plt.show()

class StandardScaler:
    """
    creation of class to save mean and std
    """

    def __init__(self, x, y):
        """
        Save statistic metrics
        """
        self.mean_x = mean(x)
        self.std_x = standard_deviation(x)
        self.mean_y = mean(y)
        self.std_y = standard_deviation(y)
    
    def normalize(self, x, y):
        """
        Apply  Z-score
        """
        norm_x = (x - self.mean_x) / self.std_x
        norm_y = (y - self.mean_y) / self.std_y
        return norm_x, norm_y
    
    def revert_coefficients(self, norm_theta0, norm_theta1):
        """
        convert theta 
        """
        theta1 = norm_theta1 * (self.std_y / self.std_x)
        theta0 = self.mean_y - theta1 * self.mean_x + norm_theta0 * self.std_y
        return theta0, theta1


def gradient_descent(mileage, price, learning_rate = 0.1, it = 1000):
    """
    Train the model by gradient descent.
    prediction 
    cost formula
    tmp_theta0 = θ0 - learning_rate * (1/m) * sum(error)
    tmp_theta1 = θ1 - learning_rate * (1/m) * sum(error) * mileage[i]
    args:
        - mileage
        - price
        - learning_rate
        - it: iteration 
    """
    m = len(mileage)
    theta0, theta1 = load_thetas()
    for _ in range(it):
        predictions = estimation(mileage, theta0, theta1)
        error = predictions - price
        tmp_theta0 = theta0 - learning_rate * (1/m) * sum(error)
        tmp_theta1 = theta1 - learning_rate * (1/m) * sum(error * mileage)
        theta0, theta1 = tmp_theta0, tmp_theta1

    return theta0, theta1

def main():
    """
    Main function of the program.
    
    Steps:
        - Load mileage and price data from CSV.
        - Normalize the values using StandardScaler.
        - Run gradient descent on normalized data.
        - Revert the normalized model to original scale.
        - Save the final theta0 and theta1.
        - Plot the regression line over real data.
    """
    try:
        mileage, price = load_data()

        scaler = StandardScaler(mileage, price)
        mileage_norm, price_norm = scaler.normalize(mileage, price)
        norm_theta0, norm_theta1 = gradient_descent(mileage_norm, price_norm, learning_rate=0.1, it = 1000)
        theta0, theta1 = scaler.revert_coefficients(norm_theta0, norm_theta1)
        save_model(theta0, theta1)
        estimation_price = estimation(mileage, theta0, theta1)
        show_graph(mileage, price, estimation_price)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    except KeyboardInterrupt:
        print("\nGood bye!")
        sys.exit(130)

if __name__ == "__main__":
    main()
 