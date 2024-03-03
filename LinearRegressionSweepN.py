import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix
from data_generator import liftDataset


def sweepN(N, d, sigma, filename):
    psfx = postfix(N, d, sigma)

    print(psfx)

    X = np.load("X" + psfx + ".npy")
    y = np.load("y" + psfx + ".npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    fr = np.arange(0.1, 1.1, 0.1)
    rmse_train_vals = []
    rmse_test_vals = []

    for f in fr:
        num_samples = int(f * X_train.shape[0])
        X_train_frac = X_train[:num_samples]
        y_train_frac = y_train[:num_samples]

        model = LinearRegression()
        model.fit(X_train_frac, y_train_frac)

        rmse_train = rmse(y_train_frac, model.predict(X_train_frac))
        rmse_test = rmse(y_test, model.predict(X_test))

        rmse_train_vals.append(rmse_train)
        rmse_test_vals.append(rmse_test)

        if f == 1.0:
            print("Model parameters:")
            print("\t Intercept: %3.5f" % model.intercept_, end="")
            for i, val in enumerate(model.coef_):
                print(", β%d: %3.5f" % (i, val), end="")
            print("\n")


    plt.plot(fr * 100, rmse_train_vals, label='Train')
    plt.plot(fr * 100, rmse_test_vals, label='Test')
    plt.xlabel('Training Samples (%)')
    plt.ylabel('RMSE')
    plt.title('Train and Test RMSE vs Sample Size')
    plt.legend()
    plt.savefig(filename)
    plt.clf()




N = 1000
sigma = 0.01
d_3 = 5
d_4 = 40

sweepN(N, d_3, sigma, "question3_3.png")
sweepN(N, d_4, sigma, "question3_4.png")




def modifiedSweepN(N, d, sigma, filename):
    psfx = postfix(N, d, sigma)

    print(psfx)

    X = liftDataset(np.load("X" + psfx + ".npy"))
    y = np.load("y" + psfx + ".npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    fr = np.arange(0.1, 1.1, 0.1)
    rmse_train_vals = []
    rmse_test_vals = []

    for f in fr:
        num_samples = int(f * X_train.shape[0])
        X_train_frac = X_train[:num_samples]
        y_train_frac = y_train[:num_samples]

        model = LinearRegression()
        model.fit(X_train_frac, y_train_frac)

        rmse_train = rmse(y_train_frac, model.predict(X_train_frac))
        rmse_test = rmse(y_test, model.predict(X_test))

        rmse_train_vals.append(rmse_train)
        rmse_test_vals.append(rmse_test)


    plt.plot(fr * 100, rmse_train_vals, label='Train')
    plt.plot(fr * 100, rmse_test_vals, label='Test')
    plt.xlabel('Training Samples (%)')
    plt.ylabel('RMSE')
    plt.title('Train and Test RMSE vs Sample Size')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

modifiedSweepN(N, d_3, sigma, "questionlifted_3.png")
modifiedSweepN(N, d_4, sigma, "questionlifted_4.png")