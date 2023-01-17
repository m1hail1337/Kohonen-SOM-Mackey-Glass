import random
import numpy as np
import pandas as pd

N = 1200
a = 0.1
b = 0.2
tau = 15


def create_sheet(array, length=10, skip_rows=0, delay=4, y_size=1) -> pd.DataFrame:  # Data to pandas sheet
    columns = np.append([f"x_{j}" for j in range(length)], [f"y_{k}" for k in range(y_size)])
    result = []

    for i in range(skip_rows, len(array) - length - y_size - delay + 1):
        x_index_start = i
        x_index_end = i + length
        y_index_start = i + length + delay
        y_index_end = i + length + delay + y_size

        row = np.append(array[x_index_start:x_index_end], array[y_index_start:y_index_end])
        result.append(row)
    result = pd.DataFrame(result, columns=columns, dtype="float")
    return result


def create_data() -> list:  # Creates one dataset
    dataset = []
    for i in range(tau):
        dataset.append(round(random.uniform(0, 1), 4))
    for j in range(tau - 1, 2000 + tau):
        dataset.append(
            round(dataset[j] - a * dataset[j] + b * dataset[j + 1 - tau] /
                  (1 + dataset[j + 1 - tau] ** 10), 4))
    return dataset


create_sheet(create_data()).to_csv(
        f"C:/Users/Mihail/PycharmProjects/NN/lab3Project/dataset")  # Enter your path to datasets package
