from ucimlrepo import fetch_ucirepo 
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X1 = wine_quality.data.features 
y1 = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 


# Features as X_Train.txt and Labels as Y_Train.txt
X2 = np.loadtxt("./human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt")

# Read labels (Y2)
Y2 = np.loadtxt("./human+activity+recognition+using+smartphones/UCI HAR Dataset/train/Y_train.txt")

# Print the shape of the loaded data to verify
print("Features shape:", X2.shape)
print("Labels shape:", Y2.shape)

def eval_gaussian(x, mu, sigma):
    """
    Evaluates the Gaussian pdf N(mu, Sigma) at each column of X.
    """
    n, N = x.shape
    C = ((2 * np.pi) ** n * np.linalg.det(sigma)) ** (-1/2)
    E = -0.5 * np.sum((x - np.tile(mu, (1, N))) @ np.linalg.inv(sigma) * (x - np.tile(mu, (1, N))), axis=0)
    g = C * np.exp(E)
    return g


