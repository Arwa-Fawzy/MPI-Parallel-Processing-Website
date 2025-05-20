from mpi4py import MPI
import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def run_linear_regression_mpi(csv_path):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df = pd.read_csv(csv_path)

    # Drop rows with missing or non-numeric data
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Ensure 'Score' exists and is numeric
    if 'Score' not in df.columns:
        raise ValueError("CSV must contain a 'Score' column")

    X = df.drop('Score', axis=1)
    y = df['Score']

    model = LinearRegression()
    model.fit(X, y)

    intercept = model.intercept_
    coef = model.coef_

    return [intercept] + list(coef)
