import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import cross_val_score
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, 
                    test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
     # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]   
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', 
                alpha=1.0, linewidths=1, marker='o', 
                s=55, label='test set')
        
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
        
#Writing a decision tree classifier for the above data    
print "Decision Tree Classification"    
from sklearn.tree import DecisionTreeClassifier
in_sample=[]
out_sample=[]
cross_val_in=[]
cross_val_out=[]
mean_in_sample=[]
mean_out_sample=[]
k_range=range(1,11)
for k in k_range:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=k)
    temp_in_sample=[]
    temp_out_sample=[]
    temp_in_sample.append(k)
    temp_out_sample.append(k)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    temp_in_sample.append(metrics.accuracy_score(y_train,y_pred_train))
    mean_in_sample.append(metrics.accuracy_score(y_train,y_pred_train))
    temp_out_sample.append(metrics.accuracy_score(y_test,y_pred_test))
    mean_out_sample.append(metrics.accuracy_score(y_test,y_pred_test))
    in_sample.append(temp_in_sample)
    out_sample.append(temp_out_sample)
    
from tabulate import tabulate
print tabulate(in_sample, headers=['Value of random_state','In-sample Accuracy Score'])
print tabulate(out_sample, headers=['Value of random_state','Out-sample Accuracy Score'])

print("Mean of In-sample:")
print(np.mean(mean_in_sample))

print("Mean of Out-sample:")
print(np.mean(mean_out_sample))

print("Standard deviation of In-sample:")
print(np.std(mean_in_sample))

print("Standard deviation of Out-sample:")
print(np.std(mean_out_sample))

print("\n\n")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=1)
tree=DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train,y_train)
cv_scores=cross_val_score(tree,X_train,y_train,cv=10)
print("CV Scores are:")    
print(cv_scores)
list_cv_scores=[]
for k in range(1,11):
    temp_list_cv_scores=[]
    temp_list_cv_scores.append(k)
    temp_list_cv_scores.append(cv_scores[k-1])
    list_cv_scores.append(temp_list_cv_scores)
    
print tabulate(list_cv_scores, headers=['Value of k', 'CV Score'])
print("Mean CV Scores are:")
print(np.mean(cv_scores))
print("Standard Deviation CV Scores are:")
print(np.std(cv_scores))
print('Accuracy of Out of sample Data:')
y_pred = tree.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))
