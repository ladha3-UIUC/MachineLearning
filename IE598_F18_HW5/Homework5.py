# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn import metrics
import seaborn as sns

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 5 - Compressing Data via Dimensionality Reduction

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).





# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Overview

# - [Unsupervised dimensionality reduction via principal component analysis 128](#Unsupervised-dimensionality-reduction-via-principal-component-analysis-128)
#   - [The main steps behind principal component analysis](#The-main-steps-behind-principal-component-analysis)
#   - [Extracting the principal components step-by-step](#Extracting-the-principal-components-step-by-step)
#   - [Total and explained variance](#Total-and-explained-variance)
#   - [Feature transformation](#Feature-transformation)
#   - [Principal component analysis in scikit-learn](#Principal-component-analysis-in-scikit-learn)
# - [Supervised data compression via linear discriminant analysis](#Supervised-data-compression-via-linear-discriminant-analysis)
#   - [Principal component analysis versus linear discriminant analysis](#Principal-component-analysis-versus-linear-discriminant-analysis)
#   - [The inner workings of linear discriminant analysis](#The-inner-workings-of-linear-discriminant-analysis)
#   - [Computing the scatter matrices](#Computing-the-scatter-matrices)
#   - [Selecting linear discriminants for the new feature subspace](#Selecting-linear-discriminants-for-the-new-feature-subspace)
#   - [Projecting samples onto the new feature space](#Projecting-samples-onto-the-new-feature-space)
#   - [LDA via scikit-learn](#LDA-via-scikit-learn)
# - [Using kernel principal component analysis for nonlinear mappings](#Using-kernel-principal-component-analysis-for-nonlinear-mappings)
#   - [Kernel functions and the kernel trick](#Kernel-functions-and-the-kernel-trick)
#   - [Implementing a kernel principal component analysis in Python](#Implementing-a-kernel-principal-component-analysis-in-Python)
#     - [Example 1 – separating half-moon shapes](#Example-1:-Separating-half-moon-shapes)
#     - [Example 2 – separating concentric circles](#Example-2:-Separating-concentric-circles)
#   - [Projecting new data points](#Projecting-new-data-points)
#   - [Kernel principal component analysis in scikit-learn](#Kernel-principal-component-analysis-in-scikit-learn)
# - [Summary](#Summary)






# # Unsupervised dimensionality reduction via principal component analysis

# ## The main steps behind principal component analysis





# ## Extracting the principal components step-by-step




df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

# ## Visualizing the important characteristics of a dataset

cols = ['Alcohol', 'Malic acid', 'Total phenols', 'Flavanoids', 'Hue']

sns.pairplot(df_wine[cols], size=2.5)
plt.tight_layout()
plt.show()


cm = np.corrcoef(df_wine[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()



# Splitting the data into 70% training and 30% test subsets.




X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)


# Standardizing the data.




sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# then $-v$ is also an eigenvector that has the same eigenvalue, since
# $$\Sigma \cdot (-v) = -\Sigma v = -\lambda v = \lambda \cdot (-v).$$
def plot_decision_regions(X, y, classifier, resolution=0.02):

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

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


lr=LogisticRegression()
lr=lr.fit(X_train_std,y_train)

y_train_pred=lr.predict(X_train_std)
print("Linear Regression on untransformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

y_test_pred=lr.predict(X_test_std)
print("Linear Regression on untransformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

svm=SVC(kernel='rbf')
svm=svm.fit(X_train_std,y_train)

y_train_pred=svm.predict(X_train_std)
print("SVM on untransformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

y_test_pred=svm.predict(X_test_std)
print("SVM on untransformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print("Plotting graph for feature 1 vs feature 2")
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# Training logistic regression classifier using the first 2 principal components.

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)



print("Plotting graph for Logistic Regression on pca-transformed training data")
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred = lr.predict(X_train_pca)
print("Logistic Regression on pca transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )


print("Plotting graph for Logistic Regression on pca-transformed testing data")
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred = lr.predict(X_test_pca)
print("Logistic Regression on pca transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

svm=SVC(kernel='rbf')
svm=svm.fit(X_train_pca,y_train)

print("Plotting graph for SVM on pca-transformed training data")
plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred=svm.predict(X_train_pca)
print("SVM on pca transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )


print("Plotting graph for SVM on pca-transformed testing data")
plot_decision_regions(X_test_pca, y_test, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred=svm.predict(X_test_pca)
print("SVM on pca transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )


# ## LDA via scikit-learn

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y_train)
X_test_lda=lda.transform(X_test_std)

print("Plotting graph for feature 1 vs feature 2")
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1])
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.show()


lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred=lr.predict(X_train_lda)
print("Linear Regression on lda transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred=lr.predict(X_test_lda)
print("Linear Regression on lda transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

svm=SVC(kernel='rbf')
svm=svm.fit(X_train_lda,y_train)

plot_decision_regions(X_train_lda, y_train, classifier=svm)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred=svm.predict(X_train_lda)
print("SVM on lda transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

plot_decision_regions(X_test_lda, y_test, classifier=svm)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred=svm.predict(X_test_lda)
print("SVM on lda transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

kpca = KernelPCA(n_components=2,kernel="rbf",gamma=0.1)
X_train_kpca=kpca.fit_transform(X_train_std,y_train)
X_test_kpca=kpca.transform(X_test_std)

print("Plotting graph for feature 1 vs feature 2")
plt.scatter(X_train_kpca[:, 0], X_train_kpca[:, 1])
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.show()

lr = LogisticRegression()
lr = lr.fit(X_train_kpca, y_train)

plot_decision_regions(X_train_kpca, y_train, classifier=lr)
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred=lr.predict(X_train_kpca)
print("Linear Regression on Kernal PCA transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

plot_decision_regions(X_test_kpca, y_test, classifier=lr)
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred=lr.predict(X_test_kpca)
print("Linear Regression on Kernel PCA transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

svm=SVC(kernel='rbf')
svm=svm.fit(X_train_kpca,y_train)

plot_decision_regions(X_train_kpca, y_train, classifier=svm)
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_train_pred=svm.predict(X_train_kpca)
print("SVM on Kernal PCA transformed training data accuracy:")
print( metrics.accuracy_score(y_train, y_train_pred) )

plot_decision_regions(X_test_kpca, y_test, classifier=svm)
plt.xlabel('KPCA 1')
plt.ylabel('KPCA 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
y_test_pred=svm.predict(X_test_kpca)
print("SVM on Kernel PCA transformed testing data accuracy:")
print( metrics.accuracy_score(y_test, y_test_pred) )

k_range=np.arange(0.1,1.0,0.1)
scores_train=[]
for k in k_range:
    temp=[]
    temp.append(k)
    kpca = KernelPCA(n_components=2,kernel="rbf",gamma=k)
    X_train_kpca=kpca.fit_transform(X_train_std,y_train)
    X_test_kpca=kpca.transform(X_test_std)
    temp.append('LR training data')
    y_train_pred=lr.predict(X_train_kpca)
    temp.append(metrics.accuracy_score(y_train,y_train_pred))
    scores_train.append(temp)
    temp=[]
    temp.append(k)
    temp.append('SVM training data')
    y_train_pred=svm.predict(X_train_kpca)
    temp.append(metrics.accuracy_score(y_train,y_train_pred))
    scores_train.append(temp)

scores_test=[]
for k in k_range:
    temp=[]
    temp.append(k)
    kpca = KernelPCA(n_components=2,kernel="rbf",gamma=k)
    X_train_kpca=kpca.fit_transform(X_train_std,y_train)
    X_test_kpca=kpca.transform(X_test_std)
    temp.append('LR testing data')
    y_test_pred=lr.predict(X_test_kpca)
    temp.append(metrics.accuracy_score(y_test,y_test_pred))
    scores_test.append(temp)
    temp=[]
    temp.append(k)
    temp.append('SVM testing data')
    y_test_pred=svm.predict(X_test_kpca)
    temp.append(metrics.accuracy_score(y_test,y_test_pred))
    scores_test.append(temp)
    
from tabulate import tabulate
print tabulate(scores_train, headers=['Value of gamma','Type of Regression Model','Accuracy Score Train Data'])
print("\n")
print tabulate(scores_test, headers=['Value of gamma','Type of Regression Model','Accuracy Score Test Data'])
    


print("\n")
print("My name is Rishabh Ladha")
print("My NetID is: ladha3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


