import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix
from data_generator import liftDataset

N = 1000
sigma = 0.01
d = 40

psfx = postfix(N, d, sigma)

X = liftDataset(np.load("X" + psfx + ".npy"))
y = np.load("y" + psfx + ".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape, "as well as %d labels." % y.shape[0])

X_train_lifted, X_test_lifted, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split lifted dataset to %d training and %d test samples" % (X_train_lifted.shape[0], X_test_lifted.shape[0]))

alphas = 2.0**np.arange(-10, 11)

cv_rmse_mean = []
cv_rmse_std = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    cv_scores = cross_val_score(model, X_train_lifted, y_train, cv=5, scoring="neg_mean_squared_error")
    cv_rmse_mean.append(np.sqrt(-np.mean(cv_scores)))
    cv_rmse_std.append(np.std(np.sqrt(-cv_scores)))

optimal_alpha = alphas[np.argmin(cv_rmse_mean)]
print("Optimal Alpha:", optimal_alpha)

plt.errorbar(np.log2(alphas), cv_rmse_mean, yerr=cv_rmse_std, marker='o')
plt.title('Cross-Validation Mean RMSE as a function of Alpha')
plt.xlabel('log2(Alpha)')
plt.ylabel('Cross-Validation Mean RMSE')
plt.show()


optimal_model = Lasso(alpha=optimal_alpha)
optimal_model.fit(X_train_lifted, y_train)

rmse_train = rmse(y_train, optimal_model.predict(X_train_lifted))
rmse_test = rmse(y_test, optimal_model.predict(X_test_lifted))

print("Train RMSE =", rmse_train)
print("Test RMSE =", rmse_test)

important_params = optimal_model.coef_[np.abs(optimal_model.coef_) > 1e-3]
intercept = optimal_model.intercept_

print("\nParameters with absolute value larger than 1e-3:")
print("Coefficients:", important_params)
print("Intercept:", intercept)
