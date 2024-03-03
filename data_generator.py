import numpy as np

def fun(x):
    """
        Compute a simple quadratic function f:R^5 -> R.

    """

    return 1.3 + 2.*x[0] - 1.1*x[1] + 0.7*x[2]  + 1.2*x[3] + \
            0.4* (x[0]**2) -1.5 *x[1]*x[3] - 0.7 * (x[4]**2) 

def postfix(N,d,sigma):
    """Converts parameters into a handy string, to be appended to file names."""
    return "_N_%d_d_%d" % (N,d) \
         + "_sig_%s" % str(sigma).replace(".","_")      

def lift(x):
    d = len(x)
    coordinates = []
    for i in range(d):
        for j in range(d):
            coordinates.append(x[i] * x[j])
    lifted = np.concatenate([x, np.array(coordinates)])
    return lifted


def liftDataset(X):
    """
    Lifts each row of the input matrix X using the lift function.

    Parameters:
    - X: Input matrix with shape (n, d), where n is the number of samples and d is the feature dimension.

    Returns:
    - X_lifted: Output matrix with shape (n, d'), where d' is the lifted feature dimension.
    """
    n, d = X.shape
    X_lifted = np.zeros((n, d + d**2))  # The lifted dimension is d + d^2
    for i in range(n):
        X_lifted[i, :] = lift(X[i, :])
    return X_lifted


if __name__ == "__main__":        
    # Set random seed for reproducibility    
    np.random.seed(42)


    # Number of samples
    N = 1000

    # Noise variance 
    sigma = 0.01


    # Feature dimension
    d = 5

    print("Generating dataset with N = %d, σ = %f, d = %d..." % (N,sigma,d), end="")

    # Generate random features
    X = np.random.randn(N, d)

    # Generate pure labels
    y = []

    for i in range(N):
        y.append(
                    fun(X[i,:])
                    )
    y = np.array(y)

    # Add noise to labels
    err = np.random.normal(scale = sigma, size = N)
    y = y + err

    print(" done")

    psfx = postfix(N,d,sigma)

    print("Saving X and y... ",end="")
    np.save("X" + psfx,X)
    np.save("y" + psfx,y)
    print(" done")


