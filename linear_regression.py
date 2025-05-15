import numpy as np
import pandas as pd


def main():
    try:
        data = pd.read_csv("data.csv")
        print(data)
        km = data["km"].values
        price = data["price"].values
        km = km.reshape(km.shape[0], 1)
        price = price.reshape(price.shape[0], 1)
        print(km.shape)
        print(price.shape)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
