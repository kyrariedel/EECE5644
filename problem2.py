import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals

np.random.seed(42)

pl0 = 0.35
pl1 = 0.65
m0 = np.array([-1, -1, -1, -1])
m1 = np.array([1, 1, 1, 1])
c0 = np.array([[5, 3, 1, -1],
                [3, 5, -2, -2],
                [1, -2, 6, 3],
                [-1, -2, 3, 4]])
c1 = np.array([[1.6, -0.5, -1.5, -1.2],
               [-0.5, 8, 6, -1.7],
               [-1.5, 6, 6, 0],
               [-1.2, -1.7, 0, 1.8]])

# Evaluate Gaussian pdf
def eval_gaussian(x, mu, Sigma):
    n, N = x.shape
    C = ((2 * np.pi) ** n * np.linalg.det(Sigma)) ** (-1/2)
    E = -0.5 * np.sum((x - np.tile(mu, (1, N))) @ np.linalg.inv(Sigma) * (x - np.tile(mu, (1, N))), axis=0)
    g = C * np.exp(E)
    return g

# Generate two multivariate Gaussian random variable objects
mean1, cov1 = m0, c0
e=eigvals(cov1)
print("Eigenvalues of cov1 are:",e)
rv1 = multivariate_normal(mean1, cov1)

mean2, cov2 = m1, c1
e=eigvals(cov2)
print("Eigenvalues of cov2 are:",e)
rv2 = multivariate_normal(mean2, cov2)

data1 = rv1.rvs(size = 100)
data2 = rv2.rvs(size = 100)

# Create a scatter plot
fig	 = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data1[:, 0], data1[:, 1], label='L = 0', alpha=0.7,c="blue")
ax.scatter(data2[:, 0], data2[:, 1], label='L = 1', alpha=0.7, c="orange")

x1, x2, x3, x4 = np.mgrid[-1:8:.01, -1:6:.01, -1:8:.01, -1:6:.01]
pos = np.dstack((x1, x2, x3, x4))

ax.contour(x1, x2, x3, x4, rv1.pdf(pos), levels=5,colors="blue")
ax.contour(x1, x2, x3, x4, rv2.pdf(pos), levels=5, colors="orange")


ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()


plt.savefig("two-gaussians.pdf", format="pdf", bbox_inches="tight")


'''
References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
https://www.w3schools.com/python/matplotlib_plotting.asp
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html

'''