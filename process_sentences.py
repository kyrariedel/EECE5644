import csv
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,confusion_matrix,roc_curve
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()



def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    s = s.strip()
    if len(s)==0:
        return s
    if not s[0].isalpha():    
        return strip_non_alpha(s[1:])         
    elif not s[-1].isalpha():       
        return strip_non_alpha(s[:-1])        
    else:
        return s

def clean(s):
    """ Create a "clean" version of a string 
    """
    return to_lower_case(strip_non_alpha(s))


# Directory of text files to be processed

directory = './SentenceCorpus/labeled_articles/'
 


# Learn the vocabulary of words in the corpus
# as well as the categories of labels used per text

categories = {}
vocabulary = {}


num_files = 0
for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    num_files +=1
    print("Processing",filename,"...",end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f,'r') as  fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                if label not in categories:
                    index = len(categories)
                    categories[label] = index
                    
                for word in words:
                    if word not in vocabulary:
                        index = len(vocabulary)
                        vocabulary[word] = index
    print(" done")            

n_words = len(vocabulary)
n_cats = len(categories)

print("Read %d files containing %d words and %d categories" % (num_files,len(vocabulary),len(categories)))

print(categories)


# Convert sentences into a "bag of words" representation.
# For example, "to be or not to be" is represented as
# a vector with length equal to the vocabulary size,
# with the value 2 at the indices corresponding to "to" and "be",
# value 1 at the indices corresponding to "or" and "not"
# and zero everywhere else. 

clf = GaussianNB()
X = []
y = []

for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    print("Converting",filename,"...",end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f,'r') as  fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                y.append(categories[label])

                features = n_words * [0]

                bag = {}
                for word in words:
                    if word not in bag:
                        bag[word] = 1
                    else:
                        bag[word] += 1
                
                for word in bag:
                    features[vocabulary[word]] = bag[word]

                X.append(features)
    print(" done")            

seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for s in seed: 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=s) #test_size = 0.2 or 0.99


    alphas = 2.0**np.arange(-15, 6)
    roc_auc_scores = []
    accuracy_scores = []
    f1_scores = []

    for alpha in alphas:
        clf.alpha = alpha
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)

    # Plot the results
    fig = plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 2)
    plt.plot(alphas, accuracy_scores, marker='o')
    plt.xscale('log')
    plt.title('Accuracy vs Smoothing Hyperparameter')
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')

    filename = 'average' + str(s) + ".pdf"
    fig.savefig(filename, bbox_inches="tight")

# Save X and y to files

with open('X_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for features in X:
        fw.writerow(features)

with open('y_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for label in y:
        fw.writerow([label])





                

