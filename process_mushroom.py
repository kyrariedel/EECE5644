import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,confusion_matrix,roc_curve
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt


# Name of file to process
filename = './mushroom/agaricus-lepiota.data'


# Learn the names of all categories present in the dataset,
# and map them to 0,1,2,...

col_maps = {}
clf = CategoricalNB()


print("Processing",filename,"...",end="")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    rows = 0
    for row in fr:
        rows += 1
        if rows == 1:
            columns = len(row)
            for c in range(columns):
                col_maps[c] = {}

        for (c,label) in enumerate(row):
            if label not in col_maps[c]:
                index = len(col_maps[c])
                col_maps[c][label] = index
print(" done")
                
print("Read %d rows having %d columns." % (rows,columns))
print("Category maps:")
for c in range(columns):
    print("\t Col %d: " % c, col_maps[c])
    


# Construct matrix X, containing the mapped 
# features, and vector y, containing the mapped
# labels.

X = []
y = []

print("Converting",filename,"...",end="")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    for row in fr:
        label = row[0]
        y.append(col_maps[0][label])

        features = []
        for (c,label) in enumerate(row[1:]):
            features.append(col_maps[c+1][label])
        
        X.append(features)

print(" done")

# Randomly split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) #test_size = 0.2 or 0.99


alphas = 2.0**np.arange(-15, 6)
roc_auc_scores = []
accuracy_scores = []
f1_scores = []


for alpha in alphas:
    clf.alpha = alpha
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc_scores.append(roc_auc)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)

# Plot the results
fig = plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(alphas, roc_auc_scores, marker='o')
plt.xscale('log')
plt.title('ROC AUC vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('ROC AUC')

plt.subplot(3, 1, 2)
plt.plot(alphas, accuracy_scores, marker='o')
plt.xscale('log')
plt.title('Accuracy vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('Accuracy')

plt.subplot(3, 1, 3)
plt.plot(alphas, f1_scores, marker='o')
plt.xscale('log')
plt.title('F1 vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('F1 Score')

fig.savefig("alpha_comparison_20.pdf", bbox_inches="tight")

# Find the alpha value that maximizes AUC
best_alpha_index = np.argmax(roc_auc_scores)
best_alpha = alphas[best_alpha_index]

print("Best alpha value:", best_alpha) 
# test_20 = 3.0517578125e-05
print("Corresponding AUC:", roc_auc_scores[best_alpha_index]) 
# test_20 = 0.9999848306953912

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) #test_size = 0.2 or 0.99

all_categories = set(X.columns)  # Assuming X is a DataFrame
missing_categories = all_categories - set(X_train.columns)
for category in missing_categories:
    X_train[category] = 0 

alphas = 2.0**np.arange(-15, 6)
roc_auc_scores_99 = []
accuracy_scores_99 = []
f1_scores_99 = []


for alpha in alphas:
    clf.alpha = alpha
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    roc_auc_scores_99.append(roc_auc)
    accuracy_scores_99.append(accuracy)
    f1_scores_99.append(f1)
         
# Plot the results
fig = plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
plt.plot(alphas, roc_auc_scores_99, marker='o')
plt.xscale('log')
plt.title('ROC AUC vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('ROC AUC')

plt.subplot(3, 1, 2)
plt.plot(alphas, accuracy_scores_99, marker='o')
plt.xscale('log')
plt.title('Accuracy vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('Accuracy')

plt.subplot(3, 1, 3)
plt.plot(alphas, f1_scores_99, marker='o')
plt.xscale('log')
plt.title('F1 vs Smoothing Hyperparameter')
plt.xlabel('alpha')
plt.ylabel('F1 Score')

fig.savefig("alpha_comparison_99.pdf", bbox_inches="tight")

# Find the alpha value that maximizes AUC
best_alpha_index = np.argmax(roc_auc_scores)
best_alpha = alphas[best_alpha_index]

print("Best alpha value:", best_alpha) 
# test_20 = 3.0517578125e-05
print("Corresponding AUC:", roc_auc_scores[best_alpha_index]) 
# test_20 = 0.9999848306953912
